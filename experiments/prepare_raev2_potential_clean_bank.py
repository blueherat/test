#!/usr/bin/env python3
"""Freeze original-image rows and re-encode a train/validation latent bank.

Selection uses only old sample protocols and Packed ImageNet metadata. The
validation rows are outside the current parameter fit, but were seen in earlier
research; they are not a previously unseen quality-confirmation set.
--prepare-only never initializes CUDA. Stage 1 is the existing official RAE.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.audit_raev2_calibration_cache import (
    DATA, DEFAULT_CONFIG, DeterministicImageNetPacked, artifact,
    install_raev2_decoder_config_compat, instantiate_from_config, load_config,
    tensor_hash,
)
from experiments.extract_raev2_decoder_audit_real_blocks import verify_source_order

RESTART = DATA / "experiments/raev2_guidance_restart_20260906"
PROTOCOL = "raev2_potential_clean_bank_fp32_v1"


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def array_hashes(values: dict[str, np.ndarray]) -> dict[str, str]:
    return {key: tensor_hash(torch.from_numpy(value)) for key, value in values.items()}


def read_protocol(directory: Path, seed: int) -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["seed"] != seed or manifest["samples"] != 5000:
        raise ValueError(f"expected the completed fixed 5000-sample bank: {directory}")
    with np.load(directory / "sample_protocol.npz", allow_pickle=False) as data:
        values = {key: data[key].copy() for key in ("sample_ids", "labels", "real_source_rows", "test_mask")}
    if any(x.shape != (5000,) for x in values.values()):
        raise ValueError("old sample protocol must contain four 5000-element arrays")
    if any(values[key].dtype != np.int64 for key in ("sample_ids", "labels", "real_source_rows")):
        raise ValueError("old protocol integer metadata must be int64")
    if values["test_mask"].dtype != np.bool_:
        raise ValueError("old protocol test mask must be bool")
    order = np.argsort(values["sample_ids"], kind="stable")
    values = {key: value[order] for key, value in values.items()}
    if not np.array_equal(values["sample_ids"], np.arange(5000)):
        raise ValueError("old protocol must contain unique global sample IDs 0..4999")
    classes, counts = np.unique(values["labels"], return_counts=True)
    if not np.array_equal(classes, np.arange(1000)) or not np.all(counts == 5):
        raise ValueError("old bank must have five samples for each of 1000 classes")
    if len(np.unique(values["real_source_rows"])) != 5000:
        raise ValueError("old real source rows contain duplicates")
    old_test_classes = np.random.default_rng(seed + 17).permutation(classes)[:200]
    if not np.array_equal(values["test_mask"], np.isin(values["labels"], old_test_classes)):
        raise ValueError("old class split differs from its fixed seed+17 rule")
    return manifest, values


def checked_identity(args, config) -> dict:
    reference = json.loads(args.identity_reference.read_text())
    params = config.stage_1.params
    if config.stage_1.target != "stage1.RAE" or params["encoder_name"] != "dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]":
        raise ValueError("expected official RAE DINOv3-L K7 encoder")
    if params["resolution"] != 256 or params["noise_tau"] != 0 or config.training.image_size != 256:
        raise ValueError("expected deterministic 256-pixel official encoding")
    if tuple(config.misc.latent_size) != (1024, 16, 16):
        raise ValueError("unexpected normalized latent shape")
    paths = {
        "config": args.config,
        "dino_weights": args.dino_ckpt_dir / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "normalization_stats": Path(params["normalization_stat_path"]),
        "packed_manifest": args.packed_data_path / "manifest.json",
    }
    identities = {key: artifact(path) for key, path in paths.items()}
    for key, actual in identities.items():
        if actual != reference[key]:
            raise ValueError(f"identity differs from the prior audited official FP32 bank: {key}")
    sources = [Path(__file__), ROOT / "experiments/audit_raev2_calibration_cache.py",
               ROOT / "experiments/extract_raev2_decoder_audit_real_blocks.py",
               ROOT / "experiments/raev2_training_core.py", ROOT / "experiments/raev2_stage1_compat.py",
               ROOT / "experiments/sample_raev2_pfr_retiming.py",
               ROOT / "experiments/run_raev2_scale_response_study.py",
               ROOT / "experiments/run_raev2_distribution_auc.py",
               ROOT / "experiments/build_imagenet_random_access.py",
               ROOT / "external/RAEv2/src/encoders/vision_encoder.py",
               ROOT / "external/RAEv2/src/encoders/models/dinov3_loader.py",
               ROOT / "external/RAEv2/src/stage1/rae.py",
               args.dino_repo_dir / "dinov3/models/vision_transformer.py"]
    source_artifacts = {str(path.resolve()): artifact(path) for path in sources}
    for old in reference["sources"].values():
        path = old["path"]
        if path in source_artifacts and source_artifacts[path] != old:
            raise ValueError(f"official encoder or preprocessing source changed: {path}")
    return {
        **identities, "identity_reference": artifact(args.identity_reference),
        "official_identity_matches_prior_audited_fp32_bank": True,
        "stage_1_params": dict(params), "sources": source_artifacts,
        "decoder_weights_instantiated_but_not_used": artifact(Path(params["pretrained_decoder_path"])),
    }


def prepare_selection(args, config) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    identity = checked_identity(args, config)
    dataset = DeterministicImageNetPacked(args.packed_data_path, split="train", image_size=256,
                                         horizontal_flip=False, index_map_path=None)
    banks, all_values = {}, {}
    try:
        packed_manifest = json.loads((args.packed_data_path / "manifest.json").read_text())
        packed_labels = np.concatenate([np.load(args.packed_data_path / record["labels_file"],
                                              allow_pickle=False) for record in packed_manifest["shards"]])
        if packed_manifest["split"] != "train" or len(packed_labels) != len(dataset):
            raise ValueError("packed label metadata does not match the train dataset")
        for name, seed in [("train", 20260801), ("validation", 20260802)]:
            bank = (args.bank_root / f"n5000_seed{seed}_scales7_v1").resolve()
            manifest, values = read_protocol(bank, seed)
            if Path(manifest["packed_data_path"]).resolve() != args.packed_data_path.resolve():
                raise ValueError("old bank and requested Packed dataset paths differ")
            source_identity = verify_source_order(manifest, dataset)
            rows = values["real_source_rows"]
            if rows.min() < 0 or rows.max() >= len(packed_labels):
                raise ValueError("old source rows outside packed dataset")
            if not np.array_equal(packed_labels[rows], values["labels"]):
                raise ValueError("old source row labels disagree with Packed metadata")
            all_values[name] = values
            banks[name] = {"seed": seed, "directory": str(bank), "manifest": artifact(bank / "manifest.json"),
                           "sample_protocol": artifact(bank / "sample_protocol.npz"), "source_identity": source_identity}
    finally:
        dataset.close()
    train = all_values["train"]
    candidates = all_values["validation"]
    eligible = ~np.isin(candidates["real_source_rows"], train["real_source_rows"])
    indices, available, skipped = [], [], []
    for label in range(1000):
        options = np.flatnonzero(candidates["labels"] == label)
        valid = options[eligible[options]]
        if not len(valid):
            raise ValueError(f"no validation source row outside train5000 for class {label}")
        indices.append(int(valid[0]))
        available.append(int(len(valid)))
        skipped.append(int(np.flatnonzero(options == valid[0])[0]))
    # Preserve ascending old global IDs, while making new bank indexing contiguous.
    indices = np.asarray(sorted(indices), dtype=np.int64)
    selected = {"train": train, "validation": {key: value[indices] for key, value in candidates.items()}}
    metadata = {name: {"ids": np.arange(len(values["sample_ids"]), dtype=np.int64),
                       "source_sample_ids": values["sample_ids"], "rows": values["real_source_rows"],
                       "labels": values["labels"], "source_test_mask": values["test_mask"]}
                for name, values in selected.items()}
    overlap = np.intersect1d(metadata["train"]["rows"], metadata["validation"]["rows"])
    if overlap.size or len(np.unique(metadata["validation"]["rows"])) != 1000:
        raise ValueError("validation rows must be unique and disjoint from train5000")
    selection = {
        "protocol": PROTOCOL, "selection_complete": True, "cuda_used": False,
        "train_rule": "All seed20260801 real_source_rows, sorted by original global sample_id; 1000 classes x 5.",
        "validation_rule": "For each class 0..999 choose the minimum seed20260802 global sample_id whose real_source_row is absent from train5000; then sort selected samples by old global ID. Error if any class has no eligible candidate.",
        "selection_uses_pixels_latents_losses_or_fid": False,
        "source_split": "ImageNet train for both banks",
        "validation_boundary": "Outside current parameter fitting only. Seed20260802 was already used in earlier research, so this is not previously unseen quality confirmation. Classes overlap fully between train and validation.",
        "old_class_test_mask": "Preserved as provenance only; the prior 800/200 class split is not used for this parameter-fit split.",
        "preprocessing": "Original Packed RGB image bytes, deterministic ADM center_crop_arr(256), FP32 pixels divided by 255, no horizontal flip, no index map. Never decoder reconstruction.",
        "identity": identity, "source_banks": banks,
        "metadata": {name: {key: value.tolist() for key, value in arrays.items()} for name, arrays in metadata.items()},
        "metadata_sha256": {name: array_hashes(arrays) for name, arrays in metadata.items()},
        "sample_counts": {"train": 5000, "validation": 1000},
        "train_validation_source_row_overlap": int(overlap.size),
        "validation_candidates_overlapping_train": int((~eligible).sum()),
        "validation_available_candidates_per_class": available,
        "validation_skipped_lower_global_ids_per_class": skipped,
        "validation_classes_requiring_later_candidate": int(np.count_nonzero(skipped)),
        "packed_label_artifacts": [artifact(args.packed_data_path / record["labels_file"]) for record in packed_manifest["shards"]],
    }
    return selection, metadata


def encode(args, config, selection: dict, metadata: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    if (output / "request.json").exists() or any(output.glob("*/latents.npy*")):
        raise FileExistsError("refusing to overwrite any prior encoding or incomplete latent array")
    os.environ["DINOV3_CKPT_DIR"] = str(args.dino_ckpt_dir.resolve())
    os.environ["DINOV3_REPO_DIR"] = str(args.dino_repo_dir.resolve())
    if not (args.dino_repo_dir / "hubconf.py").exists():
        raise FileNotFoundError("the official DINO repository must be available locally")
    install_raev2_decoder_config_compat()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    request = {
        "protocol": PROTOCOL, "selection": artifact(output / "selection.json"),
        "identity": selection["identity"], "metadata_sha256": selection["metadata_sha256"],
        "source_banks": selection["source_banks"], "validation_boundary": selection["validation_boundary"],
        "precision": "FP32 parameters, floating buffers, inputs and encoder outputs; no autocast; TF32 disabled",
        "encode_batch_size": args.batch_size, "cache_dtype": "float16", "latent_shape_per_sample": [1024, 16, 16],
        "normalization": "Exactly once inside official RAE.encode: (raw-mean)/sqrt(var+1e-5); saved bank already normalized.",
        "preprocessing": selection["preprocessing"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "cuda_device": torch.cuda.get_device_name(device),
        "torch_version": str(torch.__version__), "runner_snapshot": artifact(output / "runner_source.py"),
        "sampling": False, "decoder_queries": 0, "training": False, "fid": False,
        "consumer_rule": "Read only after root summary.json has complete=true and verify the saved metadata hashes.",
    }
    write_json(output / "request.json", request)
    started = time.perf_counter()
    dataset = DeterministicImageNetPacked(args.packed_data_path, split="train", image_size=256,
                                         horizontal_flip=False, index_map_path=None)
    rae = instantiate_from_config(config.stage_1)
    del rae.decoder
    rae = rae.float().to(device).eval().requires_grad_(False)
    if not rae.do_normalization or rae.eps != 1e-5 or rae.noise_tau != 0 or rae.training:
        raise ValueError("official encoding normalization/eval configuration mismatch")
    for tensor in list(rae.parameters()) + list(rae.buffers()) + [rae.latent_mean, rae.latent_var]:
        if tensor is not None and tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError("encoder parameters, buffers, and normalization tensors must be FP32")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    banks = {}
    calls = 0
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for name, arrays in metadata.items():
            n = len(arrays["ids"])
            directory = output / name
            temporary = directory / "latents.npy.partial"
            latent = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=(n, 1024, 16, 16))
            image_hashes, batch_sizes = [], []
            squared_roundoff = squared_encoded = max_roundoff = 0.0
            encode_started = time.perf_counter()
            for begin in range(0, n, args.batch_size):
                end = min(begin + args.batch_size, n)
                images = []
                for index in range(begin, end):
                    image, label, row = dataset[int(arrays["rows"][index])]
                    if label != int(arrays["labels"][index]) or row != int(arrays["rows"][index]):
                        raise ValueError(f"Packed image label/row mismatch in {name} at bank ID {index}")
                    if image.dtype != torch.float32 or tuple(image.shape) != (3, 256, 256):
                        raise ValueError("unexpected original-image crop precision or dimensions")
                    image_hashes.append(tensor_hash(image))
                    images.append(image)
                fresh = rae.encode(torch.stack(images).to(device)).cpu()
                calls += 1
                batch_sizes.append(end - begin)
                if fresh.dtype != torch.float32 or tuple(fresh.shape) != (end-begin, 1024, 16, 16) or not torch.isfinite(fresh).all():
                    raise FloatingPointError(f"invalid FP32 normalized encoder output in {name} at {begin}")
                half = fresh.to(torch.float16)
                if not torch.isfinite(half).all():
                    raise FloatingPointError("FP16 storage overflow")
                latent[begin:end] = half.numpy()
                error = fresh.double() - half.double()
                squared_roundoff += float(error.square().sum())
                squared_encoded += float(fresh.double().square().sum())
                max_roundoff = max(max_roundoff, float(error.abs().max()))
                if begin == 0 or end % 128 == 0 or end == n:
                    progress = {"complete": False, "bank": name, "bank_completed": end, "bank_total": n,
                                "encoder_forward_calls": calls, "bank_elapsed_seconds": time.perf_counter()-encode_started,
                                "total_elapsed_seconds": time.perf_counter()-started}
                    write_json(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            latent.flush()
            del latent
            destination = directory / "latents.npy"
            temporary.replace(destination)
            reopened = np.load(destination, mmap_mode="r", allow_pickle=False)
            if reopened.shape != (n, 1024, 16, 16) or reopened.dtype != np.float16:
                raise ValueError("serialized latent format differs from request")
            del reopened
            banks[name] = {
                "samples": n, "classes": 1000, "samples_per_class": 5 if name == "train" else 1,
                "unique_source_rows": int(len(np.unique(arrays["rows"]))),
                "latents": artifact(destination), "metadata": artifact(directory / "metadata.npz"),
                "metadata_sha256": array_hashes(arrays), "shape": [n,1024,16,16], "dtype": "float16",
                "source_image_fp32_tensor_sha256": image_hashes,
                "encoder_forward_calls": len(batch_sizes), "actual_batch_sizes": batch_sizes,
                "elapsed_seconds_including_output_hash": time.perf_counter()-encode_started,
                "fp32_to_fp16_roundoff_rms": (squared_roundoff/(n*1024*16*16))**0.5,
                "fp32_to_fp16_roundoff_relative_rms": (squared_roundoff/squared_encoded)**0.5,
                "fp32_to_fp16_roundoff_max_abs": max_roundoff,
                "fp32_encoded_rms": (squared_encoded/(n*1024*16*16))**0.5,
            }
    peak = torch.cuda.max_memory_allocated(device)
    dataset.close()
    del rae, fresh, half, error, images
    gc.collect()
    torch.cuda.empty_cache()
    summary = {
        "protocol": PROTOCOL, "complete": True, "banks": banks,
        "train_validation_source_row_overlap": 0,
        "validation_boundary": selection["validation_boundary"],
        "selection": artifact(output / "selection.json"), "request": artifact(output / "request.json"),
        "runner_snapshot": artifact(output / "runner_source.py"),
        "encoder_forward_calls": calls, "encoded_images": 6000,
        "total_elapsed_seconds": time.perf_counter()-started,
        "peak_gpu_memory_allocated_bytes": peak,
        "sampling": False, "decoder_queries": 0, "training": False, "fid": False,
        "old_banks_modified": False, "normalized_already": True,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", {"complete": True, "encoded_images": 6000, "summary": artifact(output / "summary.json")})
    print(json.dumps({"complete": True, "output": str(output), "summary": artifact(output / "summary.json"),
                      "total_elapsed_seconds": summary["total_elapsed_seconds"], "encoder_released": True}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESTART / "potential_clean_bank_fp32_v1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bank-root", type=Path, default=DATA / "experiments/raev2_ig_scale_response")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--identity-reference", type=Path, default=RESTART / "heldout_clean_c_current_fp32/request.json")
    parser.add_argument("--packed-data-path", type=Path, default=Path("/data/shared/imagenet-1k/random_access_v1"))
    parser.add_argument("--dino-ckpt-dir", type=Path, default=DATA / "models/RAEv2/encoders/dinov3")
    parser.add_argument("--dino-repo-dir", type=Path, default=DATA / "models/RAEv2/dinov3_repo")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    torch.set_num_threads(1)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config.resolve())
    selection, metadata = prepare_selection(args, config)
    if torch.cuda.is_initialized():
        raise RuntimeError("row selection must not initialize CUDA")
    selection_path = output / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError("frozen selection, source identity or runner changed")
        if artifact(output / "runner_source.py")["sha256"] != artifact(Path(__file__))["sha256"]:
            raise ValueError("runner snapshot changed")
    else:
        write_json(selection_path, selection)
        shutil.copyfile(Path(__file__), output / "runner_source.py")
        for name, arrays in metadata.items():
            directory = output / name
            directory.mkdir(exist_ok=True)
            np.savez(directory / "metadata.npz", **arrays)
    for name, arrays in metadata.items():
        with np.load(output / name / "metadata.npz", allow_pickle=False) as saved:
            if set(saved.files) != set(arrays) or any(not np.array_equal(saved[key], value) for key, value in arrays.items()):
                raise ValueError("stored metadata differs from frozen row selection")
    print(json.dumps({"selection_complete": True, "cuda_initialized": False, "selection": artifact(selection_path),
                      "sample_counts": selection["sample_counts"],
                      "source_row_overlap": selection["train_validation_source_row_overlap"],
                      "validation_classes_requiring_later_candidate": selection["validation_classes_requiring_later_candidate"]}), flush=True)
    if not args.prepare_only:
        encode(args, config, selection, metadata, output)


if __name__ == "__main__":
    main()
