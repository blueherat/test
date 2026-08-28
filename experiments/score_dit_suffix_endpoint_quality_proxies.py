#!/usr/bin/env python3
"""Score frozen DiT suffix endpoints with deliberately limited proxies.

This is a reproducible, offline audit of the 20 retained target PNGs from the
four class-207/seed-2 suffix bundles.  Two locally cached torchvision ImageNet
classifiers measure *class fidelity only*.  Their confidence is not a measure
of anatomy, topology, or structural image quality.

For the five t=60 endpoints, fixed post-hoc tail and hind-region boxes also
receive simple texture descriptors.  The boxes and descriptors were motivated
by visual inspection (especially the more naturally feathered tail in attempt
004), so they are non-general, discovery-only measurements.  They must not be
reported as a trained detector or evaluated as confirmatory evidence.

The program verifies every encoded-file and decoded-pixel hash against the
corresponding results.json, verifies the frozen target class and exact cached
weight hashes, and writes through a same-filesystem staging directory.  An
existing output directory is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.dont_write_bytecode = True

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
    convnext_tiny,
    resnet18,
)


SCHEMA_VERSION = 1
TARGET_CLASS_ID = 207
TARGET_CLASS_NAME = "golden retriever"
EXPECTED_ROLLBACKS = (60, 120, 180, 225)
EXPECTED_ATTEMPTS = tuple(range(5))
EXPECTED_IMAGE_SIZE = (256, 256)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUFFIX_ROOT = Path(
    "/home/zhoushunyu/data/eqvae/cross_scale_evidence/"
    "dit_imagenet256_suffix_repairability"
)
DEFAULT_ANNOTATION = (
    REPO_ROOT
    / "experiments/annotations/dit_imagenet256_seed2_suffix_quality_review_v2.json"
)
DEFAULT_WEIGHT_DIR = Path("/home/zhoushunyu/.cache/torch/hub/checkpoints")
DEFAULT_OUTPUT = Path(
    "/home/zhoushunyu/data/eqvae/cross_scale_evidence/"
    "dit_imagenet256_suffix_endpoint_quality_proxies/"
    "resnet18_v1_convnext_tiny_v1_posthoc_roi_v1"
)

MODEL_SPECS = {
    "resnet18_imagenet1k_v1": {
        "filename": "resnet18-f37072fd.pth",
        "sha256": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
        "bytes": 46_830_571,
        "weights": ResNet18_Weights.IMAGENET1K_V1,
        "constructor": resnet18,
    },
    "convnext_tiny_imagenet1k_v1": {
        "filename": "convnext_tiny-983f1562.pth",
        "sha256": "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
        "bytes": 114_419_221,
        "weights": ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
        "constructor": convnext_tiny,
    },
}

# Coordinates are [x0, y0, x1, y1] on the native 256x256 PNG.  They are fixed
# across the five t=60 endpoints, whose late shared prefix keeps pose aligned.
# These boxes were chosen after looking at these images and are not general.
POSTHOC_T60_ROIS = {
    "tail": (0, 52, 132, 145),
    "hind": (45, 115, 140, 250),
}

CSV_FIELDS = (
    "branch_key",
    "rollback_internal_timestep",
    "attempt",
    "role",
    "absolute_quality_v2",
    "binary_discovery_label_v2",
    "prefix_preservation_v2",
    "png_path",
    "png_sha256",
    "pixel_sha256",
    "resnet18_target_class_probability",
    "resnet18_target_class_rank",
    "resnet18_top1_class_id",
    "resnet18_top1_class_name",
    "resnet18_top1_probability",
    "convnext_tiny_target_class_probability",
    "convnext_tiny_target_class_rank",
    "convnext_tiny_top1_class_id",
    "convnext_tiny_top1_class_name",
    "convnext_tiny_top1_probability",
    "both_models_top1_target_class",
    "tail_warm_mask_fraction",
    "tail_gray_sobel_rms_all",
    "tail_gray_sobel_rms_warm_interior",
    "tail_gray_laplacian_rms_all",
    "tail_gray_laplacian_rms_warm_interior",
    "tail_gray_bandpass_rms_all",
    "tail_gray_bandpass_rms_warm_interior",
    "tail_gradient_orientation_entropy_warm_interior",
    "tail_warm_boundary_per_sqrt_area",
    "hind_warm_mask_fraction",
    "hind_gray_sobel_rms_all",
    "hind_gray_sobel_rms_warm_interior",
    "hind_gray_laplacian_rms_all",
    "hind_gray_laplacian_rms_warm_interior",
    "hind_gray_bandpass_rms_all",
    "hind_gray_bandpass_rms_warm_interior",
    "hind_gradient_orientation_entropy_warm_interior",
    "hind_warm_boundary_per_sqrt_area",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix-root", type=Path, default=DEFAULT_SUFFIX_ROOT)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--weight-dir", type=Path, default=DEFAULT_WEIGHT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="CPU is the reproducibility default; no model download is allowed.",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(image: Image.Image) -> str:
    array = np.asarray(image, dtype=np.uint8)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def finite_float(value: float | np.floating[Any]) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite numeric output: {result}")
    return result


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def branch_key(timestep: int, attempt: int) -> str:
    return f"t{timestep}_attempt{attempt:03d}"


def load_labels(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = load_json(path)
    source = payload.get("input", {})
    if source.get("class_id") != TARGET_CLASS_ID:
        raise RuntimeError(
            f"annotation class is not frozen class {TARGET_CLASS_ID}: {path}"
        )
    if source.get("class_name") != TARGET_CLASS_NAME:
        raise RuntimeError(f"unexpected annotation class name: {path}")
    reviews = payload.get("branch_reviews")
    if not isinstance(reviews, list) or len(reviews) != 20:
        raise RuntimeError(f"expected exactly 20 v2 branch reviews: {path}")
    labels: dict[str, dict[str, Any]] = {}
    for review in reviews:
        timestep = int(review["internal_timestep"])
        attempt = int(review["attempt"])
        key = branch_key(timestep, attempt)
        if key in labels:
            raise RuntimeError(f"duplicate annotation key: {key}")
        labels[key] = review
    expected = {
        branch_key(timestep, attempt)
        for timestep in EXPECTED_ROLLBACKS
        for attempt in EXPECTED_ATTEMPTS
    }
    if set(labels) != expected:
        raise RuntimeError("v2 review keys do not match the fixed 20 endpoints")
    return labels, payload


def discover_and_validate_images(
    suffix_root: Path, labels: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[Image.Image], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    bundle_records: list[dict[str, Any]] = []
    for timestep in EXPECTED_ROLLBACKS:
        bundle = (
            suffix_root
            / f"official_demo_seed2_batch0_class0207_t{timestep}_n4"
        )
        results_path = bundle / "results.json"
        results = load_json(results_path)
        if int(results.get("target_class_id", -1)) != TARGET_CLASS_ID:
            raise RuntimeError(f"bundle is not frozen class {TARGET_CLASS_ID}: {results_path}")
        if int(results.get("target_batch_index", -1)) != 0:
            raise RuntimeError(f"unexpected target batch index: {results_path}")
        if int(results.get("rollback_internal_timestep", -1)) != timestep:
            raise RuntimeError(f"rollback mismatch: {results_path}")
        branches = results.get("branches")
        if not isinstance(branches, list) or len(branches) != 5:
            raise RuntimeError(f"expected exactly five branches: {results_path}")
        seen_attempts: set[int] = set()
        for branch in branches:
            attempt = int(branch["attempt_index"])
            if attempt not in EXPECTED_ATTEMPTS or attempt in seen_attempts:
                raise RuntimeError(f"invalid or duplicate attempt in {results_path}: {attempt}")
            seen_attempts.add(attempt)
            record = branch["target_image"]
            path = bundle / str(record["relative_path"])
            encoded_hash = sha256_file(path)
            if encoded_hash != record["sha256"]:
                raise RuntimeError(f"encoded PNG hash mismatch: {path}")
            if path.stat().st_size != int(record["bytes"]):
                raise RuntimeError(f"PNG byte-count mismatch: {path}")
            with Image.open(path) as opened:
                opened.load()
                if opened.mode != record["mode"] or tuple(opened.size) != tuple(record["size"]):
                    raise RuntimeError(f"PNG mode/size mismatch: {path}")
                if opened.mode != "RGB" or tuple(opened.size) != EXPECTED_IMAGE_SIZE:
                    raise RuntimeError(f"unexpected target PNG representation: {path}")
                image = opened.copy()
            decoded_hash = pixel_sha256(image)
            if decoded_hash != record["pixel_sha256"]:
                raise RuntimeError(f"decoded pixel hash mismatch: {path}")
            if decoded_hash != branch["target_grid_tile_pixel_sha256"]:
                raise RuntimeError(f"target/grid decoded-pixel mismatch: {path}")

            key = branch_key(timestep, attempt)
            label = labels[key]
            rows.append(
                {
                    "branch_key": key,
                    "rollback_internal_timestep": timestep,
                    "attempt": attempt,
                    "role": branch["role"],
                    "absolute_quality_v2": label["absolute_quality"],
                    "binary_discovery_label_v2": label.get("binary_discovery_label"),
                    "prefix_preservation_v2": label["prefix_preservation"],
                    "png_path": str(path.resolve()),
                    "png_sha256": encoded_hash,
                    "pixel_sha256": decoded_hash,
                }
            )
            images.append(image)
        if seen_attempts != set(EXPECTED_ATTEMPTS):
            raise RuntimeError(f"missing branch attempt: {results_path}")
        bundle_records.append(
            {
                "rollback_internal_timestep": timestep,
                "results_path": str(results_path.resolve()),
                "results_sha256": sha256_file(results_path),
                "recorded_payload_sha256": results.get("payload_sha256"),
                "recorded_manifest_identity_sha256": results.get(
                    "manifest_identity_sha256"
                ),
            }
        )

    if len(rows) != 20 or len({row["png_path"] for row in rows}) != 20:
        raise RuntimeError("did not resolve exactly 20 distinct target PNG paths")
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            rows[index]["rollback_internal_timestep"], rows[index]["attempt"]
        ),
    )
    return (
        [rows[index] for index in order],
        [images[index] for index in order],
        bundle_records,
    )


def preprocessing_record(weights: Any) -> dict[str, Any]:
    transform = weights.transforms()
    interpolation = getattr(transform.interpolation, "value", str(transform.interpolation))
    return {
        "transform_repr": repr(transform),
        "resize_size": list(transform.resize_size),
        "crop_size": list(transform.crop_size),
        "mean": [finite_float(value) for value in transform.mean],
        "std": [finite_float(value) for value in transform.std],
        "interpolation": str(interpolation),
        "antialias": bool(transform.antialias),
        "input_mode": "PIL RGB",
    }


def load_models(
    weight_dir: Path, device: torch.device
) -> tuple[dict[str, tuple[torch.nn.Module, Any]], dict[str, dict[str, Any]]]:
    loaded: dict[str, tuple[torch.nn.Module, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    for name, spec in MODEL_SPECS.items():
        path = weight_dir / str(spec["filename"])
        if not path.is_file():
            raise RuntimeError(f"required local weight file is absent (downloads disabled): {path}")
        actual_hash = sha256_file(path)
        actual_bytes = path.stat().st_size
        if actual_hash != spec["sha256"] or actual_bytes != spec["bytes"]:
            raise RuntimeError(f"cached weight identity mismatch: {path}")
        weights = spec["weights"]
        categories = weights.meta.get("categories")
        if not isinstance(categories, list) or len(categories) != 1000:
            raise RuntimeError(f"unexpected ImageNet categories metadata: {name}")
        if categories[TARGET_CLASS_ID] != TARGET_CLASS_NAME:
            raise RuntimeError(f"class {TARGET_CLASS_ID} is not {TARGET_CLASS_NAME}: {name}")
        model = spec["constructor"](weights=None)
        state = torch.load(path, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"strict model state load failed: {name}")
        model.eval().requires_grad_(False).to(device)
        loaded[name] = (model, weights)
        records[name] = {
            "architecture": name.rsplit("_imagenet1k_v1", 1)[0],
            "torchvision_weight_enum": str(weights),
            "official_url_recorded_by_torchvision": weights.url,
            "local_path": str(path.resolve()),
            "local_sha256": actual_hash,
            "local_bytes": actual_bytes,
            "target_class_id_zero_based": TARGET_CLASS_ID,
            "target_class_name": categories[TARGET_CLASS_ID],
            "preprocessing": preprocessing_record(weights),
        }
    return loaded, records


def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def score_model(
    model: torch.nn.Module,
    weights: Any,
    images: Sequence[Image.Image],
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    transform = weights.transforms()
    categories = weights.meta["categories"]
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for image_batch in chunks(images, batch_size):
            inputs = torch.stack([transform(image) for image in image_batch]).to(device)
            logits = model(inputs).float().cpu()
            probabilities = torch.softmax(logits, dim=1)
            ordered = torch.argsort(probabilities, dim=1, descending=True, stable=True)
            top_probabilities, top_ids = probabilities.topk(k=5, dim=1)
            for row_index in range(probabilities.shape[0]):
                target_probability = probabilities[row_index, TARGET_CLASS_ID].item()
                target_rank = (
                    (ordered[row_index] == TARGET_CLASS_ID).nonzero(as_tuple=False).item()
                    + 1
                )
                ids = [int(value) for value in top_ids[row_index].tolist()]
                probs = [finite_float(value) for value in top_probabilities[row_index].tolist()]
                results.append(
                    {
                        "target_class_probability": finite_float(target_probability),
                        "target_class_rank": int(target_rank),
                        "top1_class_id": ids[0],
                        "top1_class_name": categories[ids[0]],
                        "top1_probability": probs[0],
                        "top5": [
                            {
                                "class_id": class_id,
                                "class_name": categories[class_id],
                                "probability": probability,
                            }
                            for class_id, probability in zip(ids, probs)
                        ],
                    }
                )
    if len(results) != len(images):
        raise RuntimeError("classifier result count mismatch")
    return results


def conv2d_same(values: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    height = kernel.shape[-2]
    width = kernel.shape[-1]
    padded = F.pad(
        values[None, None],
        (width // 2, width // 2, height // 2, height // 2),
        mode="reflect",
    )
    return F.conv2d(padded, kernel[None, None])[0, 0]


def masked_rms(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        raise RuntimeError("post-hoc ROI foreground mask is empty")
    return finite_float(torch.sqrt(torch.mean(selected.double().square())).item())


def orientation_entropy(
    gx: torch.Tensor, gy: torch.Tensor, mask: torch.Tensor, bins: int = 12
) -> float:
    magnitude = torch.sqrt(gx.square() + gy.square())
    selected_magnitude = magnitude[mask].double()
    selected_angle = torch.remainder(torch.atan2(gy[mask], gx[mask]), math.pi).double()
    total = float(selected_magnitude.sum().item())
    if total <= 0.0:
        return 0.0
    indices = torch.clamp((selected_angle / math.pi * bins).floor().long(), max=bins - 1)
    masses = torch.zeros(bins, dtype=torch.float64)
    masses.scatter_add_(0, indices, selected_magnitude)
    probabilities = masses[masses > 0.0] / masses.sum()
    entropy = -(probabilities * probabilities.log()).sum() / math.log(bins)
    return finite_float(entropy.item())


def roi_texture_descriptors(image: Image.Image, box: tuple[int, int, int, int]) -> dict[str, float]:
    rgb = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).float() / 255.0
    x0, y0, x1, y1 = box
    crop = rgb[y0:y1, x0:x1]
    if tuple(crop.shape[:2]) != (y1 - y0, x1 - x0):
        raise RuntimeError(f"invalid ROI box: {box}")
    red, green, blue = crop.unbind(dim=2)
    gray = 0.2989 * red + 0.5870 * green + 0.1140 * blue

    # A fixed color heuristic suppresses most low-saturation snow.  It is part
    # of the post-hoc descriptor definition, not an object segmentation model.
    warm = (red - blue >= 0.05) & (red - green >= -0.02) & (
        crop.amax(dim=2) - crop.amin(dim=2) >= 0.06
    )
    neighbor_count = conv2d_same(warm.float(), torch.ones((3, 3), dtype=torch.float32))
    warm_interior = warm & (neighbor_count >= 8.999)
    if int(warm_interior.sum().item()) < 64:
        raise RuntimeError(f"too few warm interior pixels in fixed ROI {box}")

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=torch.float32,
    ) / 8.0
    sobel_y = sobel_x.T.contiguous()
    laplacian_kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    gaussian_5 = torch.tensor(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ],
        dtype=torch.float32,
    ) / 256.0
    gx = conv2d_same(gray, sobel_x)
    gy = conv2d_same(gray, sobel_y)
    gradient = torch.sqrt(gx.square() + gy.square())
    laplacian = conv2d_same(gray, laplacian_kernel)
    bandpass = gray - conv2d_same(gray, gaussian_5)
    warm_boundary = warm & ~warm_interior
    warm_area = int(warm.sum().item())

    return {
        "warm_mask_fraction": finite_float(warm.float().mean().item()),
        "gray_sobel_rms_all": masked_rms(gradient, torch.ones_like(warm)),
        "gray_sobel_rms_warm_interior": masked_rms(gradient, warm_interior),
        "gray_laplacian_rms_all": masked_rms(laplacian, torch.ones_like(warm)),
        "gray_laplacian_rms_warm_interior": masked_rms(laplacian, warm_interior),
        "gray_bandpass_rms_all": masked_rms(bandpass, torch.ones_like(warm)),
        "gray_bandpass_rms_warm_interior": masked_rms(bandpass, warm_interior),
        "gradient_orientation_entropy_warm_interior": orientation_entropy(
            gx, gy, warm_interior
        ),
        "warm_boundary_per_sqrt_area": finite_float(
            int(warm_boundary.sum().item()) / math.sqrt(warm_area)
        ),
    }


def describe(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise RuntimeError("cannot summarize invalid values")
    return {
        "count": int(array.size),
        "mean": finite_float(array.mean()),
        "median": finite_float(np.median(array)),
        "minimum": finite_float(array.min()),
        "maximum": finite_float(array.max()),
    }


def classifier_comparison(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    good = [
        float(row[f"{prefix}_target_class_probability"])
        for row in rows
        if row["binary_discovery_label_v2"] == "good"
    ]
    bad = [
        float(row[f"{prefix}_target_class_probability"])
        for row in rows
        if row["binary_discovery_label_v2"] == "bad"
    ]
    good_summary = describe(good)
    bad_summary = describe(bad)
    entirely_below = max(bad) < min(good)
    entirely_above = min(bad) > max(good)
    return {
        "metric": f"{prefix}_target_class_probability",
        "meaning": "ImageNet class fidelity only; not structural quality",
        "good": good_summary,
        "bad": bad_summary,
        "bad_range_entirely_below_good_range": entirely_below,
        "bad_range_entirely_above_good_range": entirely_above,
        "ranges_overlap": not (entirely_below or entirely_above),
        "mean_bad_minus_good": finite_float(np.mean(bad) - np.mean(good)),
    }


def t60_roi_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    t60 = [row for row in rows if row["rollback_internal_timestep"] == 60]
    if len(t60) != 5:
        raise RuntimeError("expected exactly five t60 endpoints")
    result: dict[str, Any] = {
        "warning": (
            "All ROI boxes and metrics are post-hoc, pose-specific, non-general, "
            "and discovery-only. Higher texture is not synonymous with correct anatomy."
        ),
        "boxes_xyxy_native_256": {key: list(value) for key, value in POSTHOC_T60_ROIS.items()},
        "attempt004_descriptor_ranks_descending_among_five": {},
        "matched_attempt004_good_minus_attempt003_bad": {},
    }
    a4 = next(row for row in t60 if row["attempt"] == 4)
    a3 = next(row for row in t60 if row["attempt"] == 3)
    descriptor_fields = [
        field
        for field in CSV_FIELDS
        if field.startswith("tail_") or field.startswith("hind_")
    ]
    for field in descriptor_fields:
        descending = sorted(t60, key=lambda row: (-float(row[field]), int(row["attempt"])))
        rank = 1 + next(index for index, row in enumerate(descending) if row["attempt"] == 4)
        result["attempt004_descriptor_ranks_descending_among_five"][field] = rank
        result["matched_attempt004_good_minus_attempt003_bad"][field] = finite_float(
            float(a4[field]) - float(a3[field])
        )
    result["rows"] = [
        {
            "branch_key": row["branch_key"],
            **{
                field: finite_float(float(row[field]))
                for field in descriptor_fields
            },
        }
        for row in t60
    ]
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if value is None else value
                    for key, value in row.items()
                    if key in CSV_FIELDS
                }
            )
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    if not args.suffix_root.is_dir():
        raise FileNotFoundError(args.suffix_root)
    if not args.annotation.is_file():
        raise FileNotFoundError(args.annotation)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if args.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(args.device)

    labels, annotation = load_labels(args.annotation)
    rows, images, bundle_records = discover_and_validate_images(args.suffix_root, labels)
    models, model_records = load_models(args.weight_dir, device)

    scores: dict[str, list[dict[str, Any]]] = {}
    for model_name, (model, weights) in models.items():
        scores[model_name] = score_model(
            model, weights, images, device=device, batch_size=args.batch_size
        )
    for index, row in enumerate(rows):
        resnet = scores["resnet18_imagenet1k_v1"][index]
        convnext = scores["convnext_tiny_imagenet1k_v1"][index]
        row.update(
            {
                "resnet18_target_class_probability": resnet["target_class_probability"],
                "resnet18_target_class_rank": resnet["target_class_rank"],
                "resnet18_top1_class_id": resnet["top1_class_id"],
                "resnet18_top1_class_name": resnet["top1_class_name"],
                "resnet18_top1_probability": resnet["top1_probability"],
                "resnet18_top5": resnet["top5"],
                "convnext_tiny_target_class_probability": convnext[
                    "target_class_probability"
                ],
                "convnext_tiny_target_class_rank": convnext["target_class_rank"],
                "convnext_tiny_top1_class_id": convnext["top1_class_id"],
                "convnext_tiny_top1_class_name": convnext["top1_class_name"],
                "convnext_tiny_top1_probability": convnext["top1_probability"],
                "convnext_tiny_top5": convnext["top5"],
                "both_models_top1_target_class": (
                    resnet["top1_class_id"] == TARGET_CLASS_ID
                    and convnext["top1_class_id"] == TARGET_CLASS_ID
                ),
            }
        )
        if row["rollback_internal_timestep"] == 60:
            for roi_name, box in POSTHOC_T60_ROIS.items():
                for metric, value in roi_texture_descriptors(images[index], box).items():
                    row[f"{roi_name}_{metric}"] = value
        else:
            for field in CSV_FIELDS:
                if field.startswith("tail_") or field.startswith("hind_"):
                    row[field] = None

    all_resnet = all(row["resnet18_top1_class_id"] == TARGET_CLASS_ID for row in rows)
    all_convnext = all(row["convnext_tiny_top1_class_id"] == TARGET_CLASS_ID for row in rows)
    all_both = all(row["both_models_top1_target_class"] for row in rows)
    classifier_summary = {
        "resnet18": classifier_comparison(rows, "resnet18"),
        "convnext_tiny": classifier_comparison(rows, "convnext_tiny"),
    }
    neither_separates_bad = all(
        not value["bad_range_entirely_below_good_range"]
        for value in classifier_summary.values()
    )
    key_finding = (
        f"Top-1 class 207 for all 20 endpoints: ResNet18={all_resnet}, "
        f"ConvNeXt-Tiny={all_convnext}, both={all_both}. "
        + (
            "The v2 clear-bad and confident-good class-confidence ranges overlap "
            "for both networks, so these class-fidelity proxies do not distinguish "
            "the reviewed structural failures."
            if neither_separates_bad
            else "At least one class-confidence range is descriptively separated in "
            "this tiny post-hoc set; this is not evidence of structural generalization."
        )
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "dit_suffix_endpoint_quality_proxy_audit",
        "status": "posthoc_discovery_only_not_confirmatory",
        "semantic_limits": {
            "classifier_confidence": (
                "ImageNet class fidelity only. It is not structural quality, anatomy, "
                "topology, or perceptual realism."
            ),
            "t60_roi_descriptors": (
                "Post-hoc, pose-specific, non-general texture descriptors motivated "
                "by the observed feathered fur in attempt 004."
            ),
            "statistics": (
                "Descriptive comparisons only; no p-values, TPR/FPR estimates, or "
                "claims of independent samples."
            ),
        },
        "key_finding": key_finding,
        "input_validation": {
            "target_class_id_zero_based": TARGET_CLASS_ID,
            "target_class_name": TARGET_CLASS_NAME,
            "endpoint_count": len(rows),
            "all_encoded_png_hashes_match_results_records": True,
            "all_decoded_pixel_hashes_match_results_records": True,
            "all_target_grid_pixel_hashes_match": True,
            "all_pngs_rgb_256x256": True,
            "annotation_path": str(args.annotation.resolve()),
            "annotation_sha256": sha256_file(args.annotation),
            "annotation_status": annotation.get("status"),
            "bundles": bundle_records,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "device": str(device),
            "torch_num_threads": torch.get_num_threads(),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "seed": 0,
            "batch_size": args.batch_size,
        },
        "models": model_records,
        "class_fidelity_summary": {
            "all_20_top1_target_class_resnet18": all_resnet,
            "all_20_top1_target_class_convnext_tiny": all_convnext,
            "all_20_top1_target_class_both": all_both,
            "v2_binary_discovery_counts": {
                "good": sum(row["binary_discovery_label_v2"] == "good" for row in rows),
                "bad": sum(row["binary_discovery_label_v2"] == "bad" for row in rows),
                "excluded": sum(row["binary_discovery_label_v2"] is None for row in rows),
            },
            "descriptive_good_vs_bad": classifier_summary,
        },
        "posthoc_t60_roi_summary": t60_roi_summary(rows),
        "endpoints": rows,
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_dir.name}.staging-", dir=args.output_dir.parent
        )
    )
    try:
        write_csv(staging / "endpoint_scores.csv", rows)
        write_json(staging / "report.json", report)
        output_manifest = {
            "schema_version": SCHEMA_VERSION,
            "files": {
                name: {
                    "bytes": (staging / name).stat().st_size,
                    "sha256": sha256_file(staging / name),
                }
                for name in ("endpoint_scores.csv", "report.json")
            },
        }
        write_json(staging / "manifest.json", output_manifest)
        os.rename(staging, args.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(json.dumps({"output_dir": str(args.output_dir), "key_finding": key_finding}, indent=2))


if __name__ == "__main__":
    main()
