"""Extract CPU-only real-image block means aligned with saved endpoint banks.

The original scale-response runner indexes DeterministicImageNetPacked directly
with sample_protocol.real_source_rows. Its source branch reads the original
image, while the bank's decoded/real branch contains an autoencoder reconstruction.
This extractor uses the former, with the identical deterministic center crop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.raev2_training_core import DeterministicImageNetPacked


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_source_order(bank_manifest: dict, dataset: DeterministicImageNetPacked) -> dict:
    """Check the sorted-parquet identity used by the original selector/packer."""
    manifest_path = dataset.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parquet_root = Path(bank_manifest["parquet_data_path"]).resolve()
    data_dir = parquet_root / "data" if (parquet_root / "data").is_dir() else parquet_root
    if Path(manifest["source_root"]).resolve() != data_dir:
        raise ValueError("packed source_root differs from the bank's original parquet source")
    files = sorted(data_dir.glob("train-*.parquet"))
    records = manifest["shards"]
    if [path.name for path in files] != [record["source_file"] for record in records]:
        raise ValueError("packed shard order differs from sorted parquet source order")
    if dataset.index_map_path is not None or dataset._source_indices is not None:
        raise ValueError("original bank uses direct packed rows; an index map is invalid")
    for path, record in zip(files, records, strict=True):
        if path.stat().st_size != record["source_size"]:
            raise ValueError(f"source parquet size changed: {path}")
        if pq.ParquetFile(path).metadata.num_rows != record["rows"]:
            raise ValueError(f"source parquet row count changed: {path}")
    return {
        "packed_manifest": str(manifest_path),
        "packed_manifest_sha256": sha256(manifest_path),
        "source_root": str(data_dir),
        "shards_checked": len(records),
        "total_rows": len(dataset),
        "sorted_source_names_sizes_and_parquet_row_counts_match": True,
        "index_map": None,
        "original_runner": "run_raev2_scale_response_study.py: source_rows selected with seed+31, saved unchanged as real_source_rows, and directly indexed into DeterministicImageNetPacked for collect_real_latents and source-image Inception",
        "original_selector": "run_raev2_distribution_auc.py: concatenated labels from sorted train-*.parquet files; selected rows are global indices in that concatenation",
        "original_packer": "build_imagenet_random_access.py: sorted train-*.parquet, then original row-group and row order; unchanged encoded image bytes; manifest records restored to sorted source order",
        "identity_limit": "Source order, file sizes, row counts, and selected labels are checked; the full 146 GB packed payload is not rehashed against parquet in this extraction.",
    }


def selected_protocol(bank: Path, manifest: dict) -> dict[str, np.ndarray]:
    with np.load(bank / "sample_protocol.npz", allow_pickle=False) as data:
        sample_ids = data["sample_ids"].copy()
        labels = data["labels"].copy()
        source_rows = data["real_source_rows"].copy()
        test_mask = data["test_mask"].copy()
    if not (sample_ids.shape == labels.shape == source_rows.shape == test_mask.shape):
        raise ValueError("sample protocol shapes differ")
    if len(sample_ids) != manifest["samples"] or len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("invalid global sample IDs")
    classes = np.unique(labels)
    shuffled = np.random.default_rng(int(manifest["seed"]) + 17).permutation(classes)
    expected_test = np.isin(labels, shuffled[:round(len(classes) * manifest["test_fraction"])])
    if not np.array_equal(test_mask, expected_test):
        raise ValueError("saved class split differs from the original seed+17 class split")
    order = np.argsort(sample_ids, kind="stable")
    _, first = np.unique(labels[order], return_index=True)
    indices = order[np.sort(first)]
    result = {
        "sample_ids": sample_ids[indices].astype(np.int64, copy=False),
        "labels": labels[indices].astype(np.int64, copy=False),
        "source_rows": source_rows[indices].astype(np.int64, copy=False),
        "test_mask": test_mask[indices].astype(bool, copy=False),
    }
    if len(indices) != 1000 or not np.array_equal(np.sort(result["labels"]), np.arange(1000)):
        raise ValueError("expected exactly one image for each of 1000 classes")
    if not np.all(np.diff(result["sample_ids"]) > 0):
        raise ValueError("selected sample IDs must be strictly ascending")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, default=Path("/home/zhoushunyu/data/eqvae/experiments/raev2_ig_scale_response"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260801, 20260802])
    parser.add_argument("--output-dir", type=Path, default=Path("/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/decoder_linearization_real_blocks_v1"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() or any(output.glob("seed*.npz")):
        raise FileExistsError(f"refusing to overwrite extraction outputs: {output}")
    source_files = [Path(__file__).resolve()] + [REPO / "experiments" / name for name in (
        "raev2_training_core.py", "run_raev2_scale_response_study.py",
        "run_raev2_distribution_auc.py", "build_imagenet_random_access.py",
    )]
    source_hashes = {str(path): sha256(path) for path in source_files}
    request = {
        "protocol": "raev2_decoder_linearization_real_blocks_v1",
        "seeds": args.seeds,
        "selection": "per class minimum global sample_id, then ascending sample_id; saved seed+17 class split retained",
        "image_source": "original encoded RGB image in packed ImageNet; never decoded/real or latent reconstruction",
        "preprocess": "DeterministicImageNetPacked(image_size=256, horizontal_flip=False, index_map_path=None); RGB conversion and ADM center crop exactly as original bank source branch",
        "pixel_recovery": "round(255 * loader FP32 tensor) to uint8; exact division-by-255 round trip asserted for every pixel",
        "feature": "RGB uint8 / 255; 16x16 spatial grid of nonoverlapping 16x16 pixel means; [C,grid_y,grid_x] flattened to 768 (NCHW ordering); integer block sums divided in float64",
        "feature_dtype": "float64",
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "encoder_decoder_or_inception_queries": 0,
        "source_sha256": source_hashes,
        "banks": [],
    }
    summaries = []
    for seed in args.seeds:
        seed_started = time.perf_counter()
        bank = (args.bank_root / f"n5000_seed{seed}_scales7_v1").resolve()
        manifest_path = bank / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "complete" or manifest["seed"] != seed:
            raise ValueError(f"incomplete or mismatched bank: {bank}")
        dataset = DeterministicImageNetPacked(Path(manifest["packed_data_path"]), split="train", image_size=256, horizontal_flip=False)
        try:
            identity = verify_source_order(manifest, dataset)
            arrays = selected_protocol(bank, manifest)
            bank_request = {
                "seed": seed, "bank": str(bank),
                "manifest_sha256": sha256(manifest_path),
                "sample_protocol_sha256": sha256(bank / "sample_protocol.npz"),
                "source_identity": identity,
            }
            request["banks"].append(bank_request)
            atomic_json(output / "request.json", request)
            features = np.empty((len(arrays["sample_ids"]), 768), dtype=np.float64)
            pixel_digest = hashlib.sha256()
            for index, (sample_id, label, row) in enumerate(zip(arrays["sample_ids"], arrays["labels"], arrays["source_rows"], strict=True)):
                image, actual_label, actual_row = dataset[int(row)]
                if actual_label != label or actual_row != row:
                    raise ValueError(f"source identity mismatch at sample {sample_id}")
                if image.device.type != "cpu" or tuple(image.shape) != (3, 256, 256):
                    raise ValueError("expected CPU RGB 256x256 crop")
                normalized = image.numpy()
                pixels = np.rint(normalized * np.float32(255)).astype(np.uint8)
                if not np.array_equal(pixels.astype(np.float32) / np.float32(255), normalized):
                    raise ValueError("uint8 crop recovery is not exact")
                pixel_digest.update(np.ascontiguousarray(pixels).tobytes())
                sums = pixels.reshape(3, 16, 16, 16, 16).sum(axis=(2, 4), dtype=np.int64)
                features[index] = (sums.astype(np.float64) / (256 * 255)).reshape(-1)
                if (index + 1) % 200 == 0:
                    print(f"seed={seed} real crops={index + 1}/{len(features)}", flush=True)
        finally:
            dataset.close()
        if not np.isfinite(features).all() or features.min() < 0 or features.max() > 1:
            raise ValueError("invalid normalized block means")
        arrays["real_blockmean"] = features
        destination = output / f"seed{seed}.npz"
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
        with np.load(destination, allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
                raise ValueError("serialized output differs from extracted arrays")
        summaries.append({
            "seed": seed, "path": str(destination), "sha256": sha256(destination),
            "shape": list(features.shape), "dtype": str(features.dtype),
            "sample_id_min": int(arrays["sample_ids"].min()), "sample_id_max": int(arrays["sample_ids"].max()),
            "train_samples": int((~arrays["test_mask"]).sum()), "heldout_samples": int(arrays["test_mask"].sum()),
            "unique_classes": int(len(np.unique(arrays["labels"]))),
            "min": float(features.min()), "max": float(features.max()),
            "mean": float(features.mean()), "std": float(features.std()),
            "original_uint8_crops_chw_concatenated_sha256": pixel_digest.hexdigest(),
            "label_and_row_checks_passed": len(features),
            "uint8_roundtrip_checks_passed": len(features),
            "serialization_exact": True,
            "elapsed_seconds": time.perf_counter() - seed_started,
        })
    atomic_json(output / "summary.json", {
        "protocol": request["protocol"], "status": "complete", "device": "cpu",
        "outputs": summaries, "request_sha256": sha256(output / "request.json"),
        "runner_sha256": source_hashes[str(Path(__file__).resolve())],
        "elapsed_seconds": time.perf_counter() - started,
        "measurement": "original source image spatial RGB means only; no encoder, decoder, Inception, per-block standard deviation, model evaluation, or FID",
    })
    print(json.dumps({"status": "complete", "output_dir": str(output), "outputs": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
