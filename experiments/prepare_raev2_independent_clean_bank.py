#!/usr/bin/env python3
"""Freeze independent class-balanced ImageNet rows, then encode with audited FP32.

--prepare-only performs CPU metadata selection and never initializes CUDA.
Encoding reuses the frozen selection exactly; no loss, prediction or FID is read.
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


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_selection(args) -> dict:
    manifest_path = args.packed_data_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["split"] != "train":
        raise ValueError("the independent bank must use real ImageNet train rows")
    labels_paths = [args.packed_data_path / shard["labels_file"] for shard in manifest["shards"]]
    labels = np.concatenate([np.load(path, allow_pickle=False).astype(np.int64) for path in labels_paths])
    if len(labels) != manifest["total_rows"] or set(np.unique(labels)) != set(range(1000)):
        raise ValueError("packed train labels do not match the manifest and 1000-class protocol")
    exclusions = {}
    excluded_parts = []
    for name, directory in DEFAULT_BANKS.items():
        files = sorted(directory.glob("clean_rank*.npz"))
        if not files:
            raise FileNotFoundError(f"no exclusion bank in {directory}")
        parts = []
        for path in files:
            with np.load(path, allow_pickle=False) as archive:
                parts.append(archive["rows"].astype(np.int64))
        rows = np.unique(np.concatenate(parts))
        excluded_parts.append(rows)
        exclusions[name] = {
            "directory": str(directory), "source_cache_artifacts": [artifact(path) for path in files],
            "excluded_row_count": len(rows), "rows_sha256": tensor_hash(torch.from_numpy(rows)),
            "rows": rows.tolist(), "selection": "all rows including experiment IDs 1000..1023",
        }
    excluded = np.unique(np.concatenate(excluded_parts))
    if excluded.min() < 0 or excluded.max() >= len(labels):
        raise ValueError("exclusion bank row outside packed train data")
    eligible = np.ones(len(labels), dtype=bool)
    eligible[excluded] = False
    rng = np.random.Generator(np.random.PCG64(args.seed))
    rows, counts = [], []
    for label in range(1000):
        candidates = np.flatnonzero(eligible & (labels == label))
        if not len(candidates):
            raise ValueError(f"no eligible real train row for class {label}")
        counts.append(len(candidates))
        rows.append(int(rng.choice(candidates)))
    metadata = {"ids": np.arange(1000, dtype=np.int64), "rows": np.asarray(rows, dtype=np.int64),
                "labels": np.arange(1000, dtype=np.int64)}
    if len(np.unique(metadata["rows"])) != 1000 or np.intersect1d(metadata["rows"], excluded).size:
        raise ValueError("independent selection has duplicate or excluded rows")
    if not np.array_equal(labels[metadata["rows"]], metadata["labels"]):
        raise ValueError("selected row labels mismatch")
    return {
        "protocol": "raev2_independent_clean_row_selection_v1", "selection_seed": args.seed,
        "selection_rule": "For classes 0..999 in ascending order, uniformly choose one row from the ascending global packed-train rows after excluding all actual rows in A and original B; one sequential NumPy PCG64 generator.",
        "selection_uses_model_loss": False, "selection_uses_images_or_latents": False,
        "selection_uses_fid": False, "random_generator": "numpy.random.Generator(PCG64)",
        "numpy_version": np.__version__, "source_split": "train", "sample_count": 1000,
        "packed_manifest": artifact(manifest_path),
        "packed_label_artifacts": [artifact(path) for path in labels_paths],
        "exclusion_banks": exclusions, "excluded_union_row_count": len(excluded),
        "eligible_rows_per_class": counts,
        "source_metadata": {key: value.tolist() for key, value in metadata.items()},
        "source_metadata_sha256": {key: tensor_hash(torch.from_numpy(value)) for key, value in metadata.items()},
        "source_script": artifact(Path(__file__)), "cuda_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed", type=int, default=202609064)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--packed-data-path", type=Path, default=Path("/data/shared/imagenet-1k/random_access_v1"))
    parser.add_argument("--dino-ckpt-dir", type=Path, default=DATA / "models/RAEv2/encoders/dinov3")
    parser.add_argument("--dino-repo-dir", type=Path, default=DATA / "models/RAEv2/dinov3_repo")
    args = parser.parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    selection_path = out / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        if selection["selection_seed"] != args.seed:
            raise ValueError("cannot change the frozen row selection seed")
        if selection["packed_manifest"] != artifact(args.packed_data_path / "manifest.json"):
            raise ValueError("packed manifest changed after row selection")
        if selection["source_script"] != artifact(Path(__file__)):
            raise ValueError("selection script changed after freezing rows")
    else:
        selection = prepare_selection(args)
        write_json(selection_path, selection)
    metadata = {key: np.asarray(values, dtype=np.int64) for key, values in selection["source_metadata"].items()}
    for key, values in metadata.items():
        if tensor_hash(torch.from_numpy(values)) != selection["source_metadata_sha256"][key]:
            raise ValueError(f"frozen metadata hash mismatch: {key}")
    if args.prepare_only:
        print(json.dumps({"selection_complete": True, "cuda_used": False,
                          "selection": artifact(selection_path),
                          "rows_sha256": selection["source_metadata_sha256"]["rows"],
                          "excluded_union_row_count": selection["excluded_union_row_count"],
                          "sample_count": len(metadata["ids"])}), flush=True)
        return
    if (out / "request.json").exists() or (out / "clean_rank00.npz").exists():
        raise FileExistsError(f"refusing to overwrite existing encoding in {out}")
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.resolve())
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.resolve())
    install_raev2_decoder_config_compat()
    config = load_config(args.config.resolve())
    source_paths = [Path(__file__), ROOT / "experiments/audit_raev2_calibration_cache.py",
                    ROOT / "experiments/raev2_training_core.py", ROOT / "experiments/raev2_stage1_compat.py",
                    ROOT / "external/RAEv2/src/encoders/vision_encoder.py",
                    ROOT / "external/RAEv2/src/encoders/models/dinov3_loader.py",
                    ROOT / "external/RAEv2/src/stage1/rae.py",
                    args.dino_repo_dir / "dinov3/models/vision_transformer.py"]
    request = {
        "protocol": "raev2_independent_clean_bank_fp32_v1", "sample_count": 1000,
        "num_classes": 1000, "source_split": "train", "selection": artifact(selection_path),
        "selection_contents": selection, "source_metadata": selection["source_metadata"],
        "source_metadata_sha256": selection["source_metadata_sha256"],
        "heldout_means": "New train rows disjoint from all 1024 A and all 1024 original B rows; not ImageNet validation split",
        "config": artifact(args.config),
        "dino_weights": artifact(args.dino_ckpt_dir / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
        "normalization_stats": artifact(Path(config.stage_1.params["normalization_stat_path"])),
        "packed_manifest": artifact(args.packed_data_path / "manifest.json"),
        "sources": {str(path): artifact(path) for path in source_paths},
        "precision": "fp32_no_autocast_tf32_disabled", "encode_batch_size": 1,
        "horizontal_flip": False, "cache_dtype": "float16",
        "normalization_applied_by": "official RAE.encode; do not normalize saved latents again",
        "torch_version": str(torch.__version__), "sampling_performed": False, "fid_performed": False,
        "old_caches_modified": False,
    }
    write_json(out / "request.json", request)
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
    roundoff_sq = encoded_sq = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(metadata["rows"]):
            image, label, _ = dataset[int(row)]
            if int(label) != int(metadata["labels"][index]):
                raise ValueError(f"label mismatch at selected ID {index}")
            fresh = rae.encode(image.unsqueeze(0).to(device)).float().cpu()[0]
            if not torch.isfinite(fresh).all():
                raise FloatingPointError(f"nonfinite encoding at selected ID {index}")
            half = fresh.to(torch.float16)
            latents[index] = half.numpy()
            image_hashes.append(tensor_hash(image))
            roundoff_sq += float((fresh.double() - half.double()).square().sum())
            encoded_sq += float(fresh.double().square().sum())
            if index == 0 or (index + 1) % 64 == 0 or index == 999:
                progress = {"completed": index + 1, "total": 1000,
                            "elapsed_encode_seconds": time.perf_counter() - started}
                write_json(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    elapsed = time.perf_counter() - started
    peak_gpu = torch.cuda.max_memory_allocated()
    del rae, dataset, fresh, half
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"encoder_released": True, "encoding_complete": True,
                      "elapsed_encode_seconds": elapsed}), flush=True)
    cache_path = out / "clean_rank00.npz"
    np.savez_compressed(cache_path, **metadata, latents=latents)
    summary = {
        "protocol": request["protocol"], "complete": True, "samples": 1000,
        "num_classes": 1000, "unique_source_rows": 1000,
        "overlap_with_bank_a_source_rows": 0, "overlap_with_original_bank_b_source_rows": 0,
        "metadata_identical_to_frozen_selection": True,
        "selection": artifact(selection_path), "cache": artifact(cache_path),
        "keys": ["ids", "rows", "labels", "latents"], "latent_shape": list(latents.shape),
        "latent_dtype": str(latents.dtype), "source_metadata_sha256": selection["source_metadata_sha256"],
        "source_image_tensor_sha256": image_hashes, "encode_elapsed_seconds": elapsed,
        "fp32_to_half_roundoff_rms": (roundoff_sq / latents.size) ** 0.5,
        "fp32_to_half_roundoff_relative_rms": (roundoff_sq / encoded_sq) ** 0.5,
        "peak_gpu_memory_allocated_bytes": peak_gpu, "sampling_performed": False,
        "fid_performed": False, "old_caches_modified": False,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "source_image_tensor_sha256"}), flush=True)


if __name__ == "__main__":
    main()
