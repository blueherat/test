#!/usr/bin/env python3
"""Frozen DINOv3 last-CLS features for a preregistered posterior probe.

Real A and generated seed066 are training data; real C and generated seed067
are independent validation data. All four use RGB uint8, the same official
encoder preprocessing, FP32/no-TF32 backbone calls with batch one, and FP64
unit-L2 normalization. No sampling, FID, layer search, or probe fitting occurs.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "external/RAEv2/src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.audit_raev2_proximal_calibration import atomic_json, atomic_torch_save, sha256_file
from experiments.raev2_training_core import DeterministicImageNetPacked
from experiments.sample_raev2_pfr_retiming import load_config

DATA = Path("/home/zhoushunyu/data/eqvae")
RESTART = DATA / "experiments/raev2_guidance_restart_20260906"
DEFAULT_CONFIG = ROOT / "experiments/configs/raev2_strict_lpl_dinov3l_k7.yaml"
PROTOCOL = "raev2_actual_image_dinov3_cls_posterior_features_v1"


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def unit_l2_cls(raw):
    raw = raw.double()
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    if not bool(torch.isfinite(raw).all()) or bool((norm <= 0).any()):
        raise ValueError("CLS must be finite and nonzero")
    return raw / norm


def real_identities(bank):
    parts = {key: [] for key in ("ids", "rows", "labels")}
    files = sorted(bank.glob("clean_rank*.npz"))
    for path in files:
        with np.load(path) as archive:
            for key in parts:
                parts[key].append(archive[key])
    if not files:
        raise FileNotFoundError(bank)
    combined = {key: np.concatenate(value) for key, value in parts.items()}
    indices = np.flatnonzero((combined["ids"] >= 0) & (combined["ids"] < 1000))
    indices = indices[np.argsort(combined["ids"][indices])]
    selected = {key: value[indices].astype(np.int64) for key, value in combined.items()}
    if (not np.array_equal(selected["ids"], np.arange(1000))
            or not np.array_equal(selected["labels"], np.arange(1000))
            or len(np.unique(selected["rows"])) != 1000):
        raise ValueError("real bank must have one unique row per class, ids=labels=0..999")
    return selected, {"bank": str(bank.resolve()), "files": [artifact(path) for path in files],
                       "provenance": {name: artifact(bank/name) for name in ("request.json", "summary.json") if (bank/name).exists()},
                       "identities": {key: value.tolist() for key, value in selected.items()}}


def generated_artifacts(directory, expected_seed):
    request = json.loads((directory / "request.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if (request["mode"] != "official" or request["seed"] != expected_seed
            or request["sample_count"] != 1000 or request["batch_size"] != 8
            or not summary.get("complete") or summary["samples"] != 1000):
        raise ValueError("expected complete fixed-seed official 1,000-image baseline")
    source = artifact(directory / "samples.npz")
    if source["sha256"] != summary["archive_sha256"]:
        raise ValueError("generated image archive hash changed")
    return {"request": artifact(directory / "request.json"), "summary": artifact(directory / "summary.json"),
            "samples": source, "request_contents": request,
            "noise_sha256": summary["noise_sha256"], "labels_sha256": summary["labels_sha256"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--real-train-bank", type=Path, default=DATA / "experiments/spectral_ig_mechanism_v3_formal_1k/01_clean_latents")
    p.add_argument("--real-test-bank", type=Path, default=RESTART / "heldout_clean_c_current_fp32")
    p.add_argument("--fake-train", type=Path, default=RESTART / "proximal_seed202609066/official")
    p.add_argument("--fake-test", type=Path, default=RESTART / "proximal_seed202609067/official")
    p.add_argument("--packed-data", type=Path, default=Path("/data/shared/imagenet-1k/random_access_v1"))
    args = p.parse_args()
    for key, value in vars(args).copy().items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "extractor_source.py"
    snapshot.write_bytes(Path(__file__).read_bytes())
    train_ids, train_meta = real_identities(args.real_train_bank)
    test_ids, test_meta = real_identities(args.real_test_bank)
    if np.intersect1d(train_ids["rows"], test_ids["rows"]).size:
        raise ValueError("real train/test rows overlap")
    fake_train = generated_artifacts(args.fake_train, 202609066)
    fake_test = generated_artifacts(args.fake_test, 202609067)
    for key in ("config_sha256", "checkpoint_sha256", "pixel_arithmetic", "source_sha256", "precision", "time_grid"):
        if fake_train["request_contents"][key] != fake_test["request_contents"][key]:
            raise ValueError(f"generated banks differ in {key}")
    if fake_train["noise_sha256"] == fake_test["noise_sha256"]:
        raise ValueError("generated train/test noise streams must differ")
    local_repo = DATA / "models/RAEv2/dinov3_repo"
    weights_dir = DATA / "models/RAEv2/encoders/dinov3"
    weights = weights_dir / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
    if not (local_repo / "hubconf.py").is_file() or not weights.is_file():
        raise FileNotFoundError("the frozen local DINOv3 repository and weights are required")
    os.environ["DINOV3_CKPT_DIR"] = str(weights_dir.resolve())
    os.environ["DINOV3_REPO_DIR"] = str(local_repo.resolve())
    config = load_config(args.config)
    sources = [Path(__file__), ROOT / "experiments/raev2_training_core.py",
               ROOT / "experiments/audit_raev2_proximal_calibration.py",
               ROOT / "external/RAEv2/src/encoders/vision_encoder.py",
               ROOT / "external/RAEv2/src/encoders/models/dinov3_loader.py",
               local_repo / "dinov3/models/vision_transformer.py"]
    request = {"protocol": PROTOCOL, "config": artifact(args.config), "dino_weights": artifact(weights),
               "packed_manifest": artifact(args.packed_data / "manifest.json"),
               "sources": {str(path): artifact(path) for path in sources},
               "real_train": train_meta, "real_test": test_meta, "fake_train": fake_train, "fake_test": fake_test,
               "real_rows_disjoint": True, "generated_noise_disjoint": True,
               "backbone": "official RAEv2 dinov3_vitl16; final LayerNorm has affine disabled as in official encoder",
               "feature": "encoder.model.forward_features(preprocess(rgb_uint8.float()))['x_norm_clstoken']; last genuine CLS",
               "wrapper_pooled_patch_token_used": False, "feature_dimension": 1024,
               "feature_normalization": "FP64 unit L2 from the FP32 last genuine CLS; no PCA or feature selection",
               "pixels": "real Packed center_crop256 RGB uint8; generated official RGB uint8; common encoder.preprocess",
               "microbatch": 1, "precision": "FP32_no_autocast_no_TF32", "stored_unit_feature_dtype": "float64",
               "planned_probe": "fixed sum balanced BCE + 0.5*||w||^2; unpenalized bias; unit isotropic Gaussian prior; no CV",
               "planned_test": "C(f) on fixed independent C/q067; no FID/Inception selection; no test adaptation",
               "sampling_performed": False, "fid_read_or_computed": False, "torch_version": str(torch.__version__)}
    atomic_json(out / "request.json", request)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from encoders import create_encoder
    encoder = create_encoder(config.stage_1.params["encoder_name"], device=device, resolution=256)
    encoder.eval().requires_grad_(False)
    if encoder.model.norm.elementwise_affine:
        raise ValueError("official RAEv2 feature protocol requires affine-free final LayerNorm")
    dataset = DeterministicImageNetPacked(args.packed_data, split="train", image_size=256, horizontal_flip=False)
    started = time.perf_counter()
    outputs = {}
    splits = [("real_train", train_ids, None), ("fake_train", None, args.fake_train),
              ("real_test", test_ids, None), ("fake_test", None, args.fake_test)]
    with torch.inference_mode():
        for split, identities, generated in splits:
            pixels_bank = None
            if generated:
                with np.load(generated / "samples.npz") as payload:
                    pixels_bank = payload["arr_0"]
                if pixels_bank.dtype != np.uint8 or pixels_bank.shape != (1000, 256, 256, 3):
                    raise ValueError("generated pixels must be exactly 1000x256x256 RGB uint8")
            raw_parts, feature_parts, records = [], [], []
            for index in range(1000):
                if identities is not None:
                    image, label, row = dataset[int(identities["rows"][index])]
                    if int(label) != index:
                        raise ValueError("real source row/label mismatch")
                    pixels = image.mul(255).round().to(torch.uint8)
                else:
                    label, row = index, -1
                    pixels = torch.from_numpy(pixels_bank[index].copy()).permute(2, 0, 1)
                inputs = encoder.preprocess(pixels.unsqueeze(0).to(device=device, dtype=torch.float32))
                raw = encoder.model.forward_features(inputs)["x_norm_clstoken"].float().cpu()[0].contiguous()
                if raw.shape != (1024,):
                    raise ValueError("unexpected genuine CLS dimension")
                unit = unit_l2_cls(raw)
                raw_parts.append(raw)
                feature_parts.append(unit)
                records.append({"sample_id": index, "label": int(label), "source_row": int(row),
                                "rgb_uint8_sha256": tensor_hash(pixels), "preprocessed_fp32_sha256": tensor_hash(inputs),
                                "raw_cls_fp32_sha256": tensor_hash(raw), "unit_cls_fp64_sha256": tensor_hash(unit)})
                if index == 0 or (index+1) % 64 == 0 or index == 999:
                    progress = {"status": "extracting", "split": split, "completed": index+1,
                                "completed_splits": list(outputs), "elapsed_seconds": time.perf_counter()-started}
                    atomic_json(out / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
            payload = {"protocol": PROTOCOL, "split": split, "raw_cls": torch.stack(raw_parts),
                       "features": torch.stack(feature_parts), "ids": torch.arange(1000), "labels": torch.arange(1000),
                       "source_rows": torch.tensor([r["source_row"] for r in records]),
                       "records": records, "request_sha256": sha256_file(out / "request.json")}
            atomic_torch_save(out / f"{split}.pt", payload)
            outputs[split] = {**artifact(out / f"{split}.pt"), "features_sha256": tensor_hash(payload["features"]),
                              "unit_norm_max_error": float((payload["features"].norm(dim=1)-1).abs().max())}
            atomic_json(out / "split_manifest.json", outputs)
    summary = {"protocol": PROTOCOL, "complete": True, "samples_per_split": 1000,
               "splits": outputs, "elapsed_seconds": time.perf_counter()-started,
               "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
               "sampling_performed": False, "fid_read_or_computed": False}
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "progress.json", {**summary, "status": "complete"})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
