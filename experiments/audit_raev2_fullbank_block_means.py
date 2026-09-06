#!/usr/bin/env python3
"""CPU-only complete-bank RGB block-mean prerequisite check, no model calls.

Uses original real crops and already clamped, BF16-decoded images stored in
FP16. This is a different numerical path from the FP32 decoder JVP audit.
All 5000 paired observations and the original class split are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.extract_raev2_decoder_audit_real_blocks import (
    DeterministicImageNetPacked, atomic_json, sha256, verify_source_order,
)
from experiments.audit_raev2_proximal_calibration import write_csv

PROTOCOL = "raev2_complete_bank_fixed_blockmean_prerequisite_v1"
SCALES = ("scale_s1p000000", "scale_s1p780000")
DEFAULT_BANKS = Path("/home/zhoushunyu/data/eqvae/experiments/raev2_ig_scale_response")
DEFAULT_OUTPUT = Path("/home/zhoushunyu/data/eqvae/experiments/raev2_guidance_restart_20260906/decoder_fullbank_blockmeans_v1")


def protocol_arrays(bank, manifest):
    with np.load(bank / "sample_protocol.npz", allow_pickle=False) as archive:
        arrays = {"sample_ids": archive["sample_ids"].astype(np.int64),
                  "labels": archive["labels"].astype(np.int64),
                  "source_rows": archive["real_source_rows"].astype(np.int64),
                  "test_mask": archive["test_mask"].astype(bool)}
    if any(value.shape != (5000,) for value in arrays.values()):
        raise ValueError("all protocol identity arrays must have 5000 entries")
    if not np.array_equal(arrays["sample_ids"], np.arange(5000)):
        raise ValueError("source IDs must be the complete ordered 0..4999 cohort")
    classes, counts = np.unique(arrays["labels"], return_counts=True)
    if not np.array_equal(classes, np.arange(1000)) or not np.all(counts == 5):
        raise ValueError("expected five samples for each of 1000 classes")
    shuffled = np.random.default_rng(int(manifest["seed"])+17).permutation(classes)
    expected = np.isin(arrays["labels"], shuffled[:round(len(classes)*manifest["test_fraction"])])
    if not np.array_equal(arrays["test_mask"], expected) or expected.sum() != 1000:
        raise ValueError("source split differs from the original 4000/1000 class-disjoint split")
    return arrays


def analyze(arrays):
    real, full, guided = (arrays[key] for key in ("real_blockmean", "full_blockmean", "guided_blockmean"))
    train, heldout = ~arrays["test_mask"], arrays["test_mask"]
    labels = arrays["labels"]
    variance = real[train].var(axis=0, ddof=1)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("nonpositive training real variance; no added ridge is permitted")
    result, class_rows = {}, []
    for name, scale in (("unwhitened", np.ones(768)), ("train_real_diagonal_whitened", np.sqrt(variance))):
        train_error = (full[train].mean(0)-real[train].mean(0))/scale
        axis_norm = np.linalg.norm(train_error)
        if not np.isfinite(axis_norm) or axis_norm == 0:
            raise ValueError("training error axis is zero or nonfinite")
        axis = train_error/axis_norm
        splits = {}
        for split, mask in (("train", train), ("heldout", heldout)):
            error = (full[mask].mean(0)-real[mask].mean(0))/scale
            actual = (guided[mask].mean(0)-full[mask].mean(0))/scale
            before, after = float(error @ error), float((error+actual) @ (error+actual))
            splits[split] = {"samples": int(mask.sum()), "classes": int(len(np.unique(labels[mask]))),
                            "squared_mean_error_full": before, "squared_mean_error_ig": after,
                            "ig_minus_full_squared_mean_error": after-before,
                            "relative_reduction": 1-after/before if before > 0 else None,
                            "actual_mean_shift_norm": float(np.linalg.norm(actual)),
                            "linear_cross_term": float(2*error @ actual),
                            "mean_shift_squared_norm": float(actual @ actual)}
        projection = ((guided[heldout]-full[heldout])/scale) @ axis
        heldout_labels = labels[heldout]
        class_means = []
        for label in np.unique(heldout_labels):
            values = projection[heldout_labels == label]
            if len(values) != 5:
                raise ValueError("a held-out class does not contain exactly five samples")
            class_means.append(values.mean())
            class_rows.append({"scaling": name, "label": int(label), "samples": 5,
                               "heldout_actual_projection_onto_train_error_axis": float(values.mean())})
        class_means = np.asarray(class_means, dtype=np.float64)
        if len(class_means) != 200 or not np.isfinite(class_means).all():
            raise ValueError("expected 200 finite held-out class means")
        mean, se = float(class_means.mean()), float(class_means.std(ddof=1)/np.sqrt(200))
        result[name] = {"training_error_axis_norm": float(axis_norm), "empirical_squared_mean_error": splits,
                        "heldout_actual_projection_onto_train_error_axis": {
                            "classes": 200, "images_per_class": 5, "mean": mean,
                            "class_mean_standard_error": se,
                            "class_mean_normal_95_descriptive": [mean-1.96*se, mean+1.96*se],
                            "fraction_class_means_negative": float((class_means < 0).mean()),
                            "sign": "negative points opposite the training Full-minus-real mean-error axis"}}
    return result, class_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started, cpu_started = time.perf_counter(), time.process_time()
    out = args.output_dir.expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    source_paths = (Path(__file__), ROOT / "experiments/extract_raev2_decoder_audit_real_blocks.py",
                    ROOT / "experiments/raev2_training_core.py",
                    ROOT / "experiments/run_raev2_scale_response_study.py",
                    ROOT / "experiments/run_raev2_distribution_auc.py",
                    ROOT / "experiments/build_imagenet_random_access.py")
    request = {"protocol": PROTOCOL, "seeds": [20260801, 20260802],
               "bank_root": str(args.bank_root.resolve()), "output_dir": str(out),
               "cohort": "all 5000 per seed, no subsampling; original 4000 train / 1000 heldout image split",
               "classes": "1000 total, five images per class; 800 train / 200 heldout classes",
               "real_source": "original packed RGB image with original deterministic ADM center crop, never reconstruction",
               "cached_generated_source": "BF16 decoder output clamped to [0,1], stored float16 NHWC; no new decode",
               "feature": "768 linear RGB block means: 16x16 grid of 16x16 pixels, RGB/NCHW flatten",
               "real_arithmetic": "exact uint8 recovery from loader; integer block sums / (256*255) in float64",
               "cache_arithmetic": "original FP16 cache values averaged in float64; no uint8 quantization",
               "whitening": "diagonal std from 4000 training real images only; no ridge",
               "projection_axis": "normalized Full-minus-real feature-mean error on training classes only",
               "interval": "descriptive normal interval across 200 heldout class means; never across 1000 individual images",
               "device": "cpu", "torch_threads": 1, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
               "numerical_boundary": "This FP16 cached BF16 clamped path differs from the FP32 JVP audit; no direct derivative attribution is inferred.",
               "model_calls": 0, "training": False, "sampling": False, "fid": False, "banks": []}
    atomic_json(out / "request.json", request)
    (out / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    summaries = []
    try:
        for seed in request["seeds"]:
            seed_started = time.perf_counter()
            bank = (args.bank_root / f"n5000_seed{seed}_scales7_v1").resolve()
            manifest = json.loads((bank / "manifest.json").read_text())
            if (manifest["status"] != "complete" or manifest["seed"] != seed
                    or manifest["world_size"] != 4 or manifest["precision"] != "bf16"
                    or manifest["samples"] != 5000 or manifest["state_key"] != "ema"
                    or manifest["sampler_steps"] != 100 or manifest["ig_interval"] != [.1, 1.]
                    or not manifest["same_noise_and_labels_across_scales"]):
                raise ValueError("source bank protocol differs from the frozen completed experiment")
            arrays = protocol_arrays(bank, manifest)
            dataset = DeterministicImageNetPacked(Path(manifest["packed_data_path"]), split="train",
                                                  image_size=256, horizontal_flip=False)
            maps, cache_records = {}, []
            for scale in SCALES:
                for rank in range(4):
                    path = bank / "decoded" / f"{scale}_rank{rank:02d}.npy"
                    value = np.load(path, mmap_mode="r", allow_pickle=False)
                    if value.dtype != np.float16 or value.shape != (1250, 256, 256, 3):
                        raise ValueError(f"incorrect cached decoder array {path}")
                    maps[scale, rank] = value
                    stat = path.stat()
                    cache_records.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            try:
                bank_request = {"seed": seed, "bank": str(bank), "manifest_sha256": sha256(bank / "manifest.json"),
                                "sample_protocol_sha256": sha256(bank / "sample_protocol.npz"),
                                "source_identity": verify_source_order(manifest, dataset), "cache_files": cache_records,
                                "original_checkpoint": {key: manifest[key] for key in ("checkpoint", "checkpoint_size", "state_key")}}
                request["banks"].append(bank_request)
                atomic_json(out / "request.json", request)
                features = {name: np.empty((5000, 768), dtype=np.float64)
                            for name in ("real_blockmean", "full_blockmean", "guided_blockmean")}
                digests = {name: hashlib.sha256() for name in ("real_uint8_chw", *SCALES)}
                for index, (sample_id, label, row) in enumerate(zip(arrays["sample_ids"], arrays["labels"], arrays["source_rows"], strict=True)):
                    image, actual_label, actual_row = dataset[int(row)]
                    if actual_label != label or actual_row != row or tuple(image.shape) != (3, 256, 256) or image.device.type != "cpu":
                        raise ValueError(f"source crop identity mismatch at global id {sample_id}")
                    normalized = image.numpy()
                    pixels = np.rint(normalized*np.float32(255)).astype(np.uint8)
                    if not np.array_equal(pixels.astype(np.float32)/np.float32(255), normalized):
                        raise ValueError("loader pixels did not exactly roundtrip through uint8")
                    digests["real_uint8_chw"].update(np.ascontiguousarray(pixels).tobytes())
                    sums = pixels.reshape(3, 16, 16, 16, 16).sum(axis=(2, 4), dtype=np.int64)
                    features["real_blockmean"][index] = (sums.astype(np.float64)/(256*255)).reshape(-1)
                    for scale, name in zip(SCALES, ("full_blockmean", "guided_blockmean")):
                        cached = np.asarray(maps[scale, int(sample_id)%4][int(sample_id)//4])
                        if not np.isfinite(cached).all() or cached.min() < 0 or cached.max() > 1:
                            raise ValueError(f"invalid cached pixels at {scale} global id {sample_id}")
                        digests[scale].update(cached.tobytes())
                        means = cached.transpose(2, 0, 1).reshape(3, 16, 16, 16, 16).mean(axis=(2, 4), dtype=np.float64)
                        features[name][index] = means.reshape(-1)
                    if (index+1) % 200 == 0:
                        progress = {"status": "extracting", "seed": seed, "completed": index+1,
                                    "total": 5000, "elapsed_seconds": time.perf_counter()-started, "model_calls": 0}
                        atomic_json(out / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
            finally:
                dataset.close()
            for record in cache_records:
                current = Path(record["path"]).stat()
                if (current.st_size, current.st_mtime_ns) != (record["size"], record["mtime_ns"]):
                    raise ValueError("a cached array changed during extraction")
            if any(not np.isfinite(value).all() for value in features.values()):
                raise FloatingPointError("nonfinite block means")
            arrays.update(features)
            path = out / f"seed{seed}.npz"
            temporary = path.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            temporary.replace(path)
            with np.load(path, allow_pickle=False) as saved:
                if set(saved.files) != set(arrays) or any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
                    raise ValueError("serialized complete bank differs from extraction")
            analysis, class_rows = analyze(arrays)
            atomic_json(out / f"seed{seed}_analysis.json", analysis)
            write_csv(out / f"seed{seed}_heldout_class_projections.csv", class_rows)
            summaries.append({"seed": seed, "path": str(path), "sha256": sha256(path),
                              "samples": 5000, "train_samples": 4000, "heldout_samples": 1000,
                              "heldout_classes": 200, "features_shape": [5000, 768],
                              "pixel_content_sha256_global_id_order": {key: value.hexdigest() for key, value in digests.items()},
                              "pixel_hash_encoding": "real uint8 CHW; cached float16 NHWC; all 5000 row bytes in ascending global ID order",
                              "serialization_exact": True, "source_row_and_label_checks": 5000,
                              "elapsed_seconds": time.perf_counter()-seed_started, "analysis": analysis})
            atomic_json(out / "partial_summary.json", {"status": "running", "outputs": summaries})
        summary = {"protocol": PROTOCOL, "complete": True, "device": "cpu", "model_calls": 0,
                   "real_images_read": 10000, "cached_images_read": 20000,
                   "outputs": summaries, "request_sha256": sha256(out / "request.json"),
                   "total_wall_seconds": time.perf_counter()-started, "cpu_seconds": time.process_time()-cpu_started,
                   "max_resident_memory_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "interpretation": "same fixed linear feature and existing full banks; retrospective prerequisite diagnostic, not FID or a new guidance result; class intervals descriptive only"}
        atomic_json(out / "summary.json", summary)
        atomic_json(out / "progress.json", {"status": "complete", "complete": True,
                    "seeds": request["seeds"], "total_wall_seconds": summary["total_wall_seconds"], "model_calls": 0})
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    except BaseException as exc:
        atomic_json(out / "progress.json", {"status": "failed", "error": repr(exc),
                    "completed_seeds": [item["seed"] for item in summaries], "elapsed_seconds": time.perf_counter()-started})
        raise


if __name__ == "__main__":
    main()
