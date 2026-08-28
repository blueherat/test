#!/usr/bin/env python3
"""Extract strictly label-free visual tracks from preterminal DiT predictions.

The input is a collection of completed observation-only custom DiT trace
bundles.  It can be specified directly, or indirectly through the
``source_inventory.json`` of a label-free primary/posterior analysis.  No
review, label, candidate score, calibration threshold, alert, or intervention
artifact is accepted or opened.

At fixed sampling checkpoints, ``pred_xstart`` is decoded by the pinned local
``stabilityai/sd-vae-ft-mse`` snapshot in FP32.  The resulting float image is
kept in memory only.  Hand-designed blur/edge tracks and fixed ImageNet
ResNet-18 semantic/feature tracks are then measured.  The final directory is
published atomically and contains only label-free scalar features, time series,
their catalog/formulas, provenance, and manifest/completion hashes.

These measurements are diagnostics, not posterior probabilities or calibrated
quality scores.  Every level value is measurable after the current model call
and before that step's innovation; a jump becomes measurable when its later
endpoint has been produced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# Enforce offline model loading and avoid modifying frozen source directories.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.analyze_dit_bad_good_custom_traces import (  # noqa: E402
    IMAGE_SIZE,
    STEPS,
    TraceRecord,
    _parse_csv_ints,
    _require_regular,
    atomic_json_dump,
    canonical_sha256,
    discover_trace_dirs,
    load_json,
    load_validated_trace,
    sha256_array,
    sha256_file,
)
from experiments.reproduce_dit_imagenet256 import (  # noqa: E402
    VAE_REVISION,
    VAE_SCALING_FACTOR,
    validate_vae_snapshot,
)


SCHEMA_VERSION = 1
EXPERIMENT = "dit_predxstart_preterminal_visual_tracks_label_free"
DEFAULT_CHECKPOINTS = tuple(range(69, 150, 10))
DEFAULT_INTERNAL_TIMESTEPS = tuple(STEPS - 1 - k for k in DEFAULT_CHECKPOINTS)
GRID_SIZE = 4
ACTIVE_TILE_COUNT = 8
EDGE_SHIFT_RADIUS = 4
CAM_SHIFT_RADIUS = 1
EPS = 1e-12
RESNET18_FILENAME = "resnet18-f37072fd.pth"
RESNET18_BYTES = 46_830_571
RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
IDENTIFIER_COLUMNS = (
    "sample_index",
    "run_index",
    "global_seed",
    "class_slot",
    "class_id",
    "trace_dir",
    "endpoint_png_path",
)
LEVEL_TRACKS = (
    "decoded_local_blur_severity",
    "decoded_edge_tangle",
    "resnet18_target_log_odds",
    "decoder_clipping_fraction",
)
JUMP_TRACKS = (
    "decoded_coherent_edge_jump",
    "resnet18_embedding_cosine_jump",
    "resnet18_target_log_odds_drop",
    "resnet18_target_cam_jump",
)
FORBIDDEN_INPUT_TOKENS = (
    "review",
    "annotation",
    "consensus",
    "label",
    "candidate_score",
    "calibration",
    "threshold",
    "alert",
)
ALLOWED_SOURCE_EXPERIMENTS = {
    "dit_bad_good_custom_trace_metric_discovery",
    "dit_targeted_posterior_evidence_label_free",
}


PROTOCOL: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "experiment": EXPERIMENT,
    "status": "LABEL_FREE_OBSERVATION_ONLY",
    "checkpoint_default": {
        "sampling_steps": list(DEFAULT_CHECKPOINTS),
        "internal_timesteps": list(DEFAULT_INTERNAL_TIMESTEPS),
        "meaning": (
            "pred_xstart after the current model evaluation and before the current "
            "reverse-step innovation; the default adjacent spacing is ten steps"
        ),
        "latest_allowed_sampling_step": 149,
    },
    "decode": {
        "formula": "I_k=clip((VAE(pred_xstart_k/0.18215)+1)/2,0,1)",
        "vae": "pinned local stabilityai/sd-vae-ft-mse revision",
        "precision": "FP32 only",
        "quantization": "none",
        "saved_images": False,
        "model_load_count": "one per process",
        "offline_only": True,
    },
    "pixel_features": {
        "gray": "Y=0.2989*R+0.5870*G+0.1140*B",
        "derivatives": (
            "Gaussian(Y,sigma=0.7), normalized 3x3 Sobel gx/gy, and discrete "
            "Laplacian l"
        ),
        "active_tiles": (
            "fixed 4x4 equal tiles; the eight tiles with greatest grayscale "
            "variance, with stable index tie breaking"
        ),
        "local_blur": (
            "q_j=mean(l_j^2)/(mean(g_j^2)+1e-12); "
            "B=-log(percentile_25(q over active tiles)+1e-12)"
        ),
        "edge_tangle": (
            "J=Gaussian_sigma1.5([[gx^2,gx*gy],[gx*gy,gy^2]]); "
            "c=sqrt((Jxx-Jyy)^2+4Jxy^2)/(Jxx+Jyy+eps); each tile has "
            "sum(tr(J)*(1-c))/(sum(tr(J))+eps); output is percentile_90 over "
            "the same eight active tiles"
        ),
        "coherent_edge_jump": (
            "E=g*c, globally L2-normalized; for consecutive selected checkpoints, "
            "1-max inner_product(E_current, shifted(E_previous)), shifts +/-4 pixels"
        ),
        "decoder_clipping_fraction": (
            "fraction of decoded normalized RGB values outside [0,1] before clipping"
        ),
    },
    "resnet18_features": {
        "model": "torchvision ResNet-18 ImageNet-1K V1 architecture",
        "weights": RESNET18_FILENAME,
        "input": (
            "center crop decoded 256x256 float RGB to 224x224, then ImageNet V1 "
            "channel normalization"
        ),
        "embedding_jump": "1-cosine of consecutive avgpool embeddings",
        "target_log_odds": (
            "target class logit minus logsumexp of the other 999 logits; this is "
            "not a quality posterior"
        ),
        "target_drop": "max(previous target log-odds-current target log-odds,0)",
        "target_cam": (
            "relu(sum_c fc_weight[target,c]*layer4_feature[c]), L2-normalized on 7x7"
        ),
        "target_cam_jump": (
            "1-max inner product of current CAM and previous CAM shifted +/-1 cell"
        ),
    },
    "supervision_policy": {
        "labels_read_or_emitted": False,
        "reviews_read": False,
        "candidate_scores_read": False,
        "calibration_thresholds_read": False,
        "alerts_read": False,
        "auc_or_selection_computed": False,
    },
}


@dataclass(frozen=True)
class Extraction:
    record: TraceRecord
    tracks: dict[str, np.ndarray]


def _canonical_self_hash(payload: Mapping[str, Any], key: str) -> str:
    copied = dict(payload)
    copied.pop(key, None)
    return canonical_sha256(copied)


def _array_record(value: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "raw_sha256": sha256_array(value),
    }


def _assert_clean_input_name(path: Path, description: str) -> None:
    lowered = path.name.lower()
    if any(token in lowered for token in FORBIDDEN_INPUT_TOKENS):
        raise RuntimeError(f"{description} name looks supervised/forbidden: {path}")


def _validate_label_free_source_inventory(
    path: Path,
) -> tuple[list[Path], dict[Path, dict[str, Any]]]:
    """Validate an analysis envelope and return only its bound trace roots."""

    path = path.expanduser().absolute()
    _require_regular(path, "label-free source inventory")
    path = path.resolve()
    if path.name != "source_inventory.json":
        raise RuntimeError("--source-inventory must name source_inventory.json")
    parent = path.parent
    manifest_path = parent / "manifest.json"
    completion_path = parent / "completion.json"
    _require_regular(manifest_path, "source-analysis manifest")
    _require_regular(completion_path, "source-analysis completion")
    manifest = load_json(manifest_path)
    completion = load_json(completion_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("experiment") not in ALLOWED_SOURCE_EXPERIMENTS
        or completion.get("complete") is not True
    ):
        raise RuntimeError(f"source analysis is not complete: {parent}")
    if manifest.get("source_inventory_sha256") != sha256_file(path):
        raise RuntimeError(f"source inventory is not bound by its manifest: {path}")
    if completion.get("manifest_file_sha256") != sha256_file(manifest_path):
        raise RuntimeError(f"source completion does not bind manifest bytes: {parent}")
    manifest_identity = dict(manifest)
    recorded_identity = manifest_identity.pop("identity_sha256", None)
    if recorded_identity != canonical_sha256(manifest_identity):
        # Older extractors hash the manifest including all fields except only the
        # as-yet-unset identity.  The expression above is that exact convention.
        raise RuntimeError(f"source manifest identity is invalid: {parent}")
    if completion.get("manifest_identity_sha256") != recorded_identity:
        raise RuntimeError(f"source completion/manifest identity differs: {parent}")

    inventory = load_json(path)
    locked = inventory.get("locked_consensus")
    if locked is not None:
        raise RuntimeError("source inventory carries a non-null locked consensus")
    runs = inventory.get("trace_runs")
    ordered_seeds = inventory.get("ordered_seeds")
    ordered_classes = inventory.get("ordered_classes")
    if (
        not isinstance(runs, list)
        or not runs
        or not all(isinstance(item, dict) for item in runs)
        or not isinstance(ordered_seeds, list)
        or not isinstance(ordered_classes, list)
    ):
        raise RuntimeError(f"malformed label-free source inventory: {path}")
    roots: list[Path] = []
    bindings: dict[Path, dict[str, Any]] = {}
    observed_seeds: list[int] = []
    for item in runs:
        root_raw = item.get("root")
        seed = item.get("global_seed")
        if not isinstance(root_raw, str) or type(seed) is not int:
            raise RuntimeError(f"malformed trace record in source inventory: {path}")
        root = Path(root_raw).expanduser().absolute().resolve()
        _assert_clean_input_name(root, "trace root")
        required_binding_keys = (
            "identity_sha256",
            "manifest_sha256",
            "completion_sha256",
            "trace_sha256",
            "scientific_fingerprint_sha256",
        )
        if any(not isinstance(item.get(key), str) for key in required_binding_keys):
            raise RuntimeError(f"incomplete trace binding in source inventory: {root}")
        bindings[root] = {
            key: item[key] for key in required_binding_keys
        } | {
            "global_seed": seed,
            "classes": tuple(ordered_classes),
        }
        roots.append(root)
        observed_seeds.append(seed)
    if observed_seeds != ordered_seeds or len(set(roots)) != len(roots):
        raise RuntimeError(f"source inventory run order or uniqueness changed: {path}")
    return roots, bindings


def _collect_trace_dirs(
    args: argparse.Namespace,
) -> tuple[list[Path], list[dict[str, Any]], dict[Path, dict[str, Any]]]:
    paths: list[Path] = []
    inventories: list[dict[str, Any]] = []
    bindings: dict[Path, dict[str, Any]] = {}
    for inventory_path in args.source_inventory or []:
        inventory_path = inventory_path.expanduser().absolute().resolve()
        roots, inventory_bindings = _validate_label_free_source_inventory(inventory_path)
        paths.extend(roots)
        overlap = set(bindings) & set(inventory_bindings)
        if overlap:
            raise RuntimeError(f"trace roots repeated by source inventories: {sorted(overlap)}")
        bindings.update(inventory_bindings)
        inventories.append(
            {
                "path": str(inventory_path),
                "sha256": sha256_file(inventory_path),
                "manifest_sha256": sha256_file(inventory_path.parent / "manifest.json"),
                "completion_sha256": sha256_file(
                    inventory_path.parent / "completion.json"
                ),
                "trace_run_count": len(roots),
            }
        )
    if args.trace_dir or args.trace_root:
        paths.extend(discover_trace_dirs(args))
    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        path = path.expanduser().absolute().resolve()
        _assert_clean_input_name(path, "trace root")
        if path in seen:
            raise RuntimeError(f"duplicate trace directory from inputs: {path}")
        seen.add(path)
        resolved.append(path)
    if not resolved:
        raise RuntimeError("no trace directories selected")
    return resolved, inventories, bindings


def _check_inventory_trace_binding(
    record: TraceRecord, binding: Mapping[str, Any] | None
) -> None:
    if binding is None:
        return
    expected = {
        "identity_sha256": record.identity_sha256,
        "manifest_sha256": record.manifest_sha256,
        "completion_sha256": record.completion_sha256,
        "trace_sha256": record.trace_sha256,
        "scientific_fingerprint_sha256": record.scientific_fingerprint_sha256,
        "global_seed": record.global_seed,
        "classes": record.classes,
    }
    if any(binding.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"source inventory trace binding changed: {record.root}")


def _validate_resnet_weights(path: Path) -> dict[str, Any]:
    path = path.expanduser().absolute()
    _require_regular(path, "pinned ResNet-18 weights")
    path = path.resolve()
    if path.name != RESNET18_FILENAME:
        raise RuntimeError(f"wrong ResNet-18 checkpoint filename: {path.name}")
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != RESNET18_BYTES or digest != RESNET18_SHA256:
        raise RuntimeError(
            f"ResNet-18 checkpoint identity differs: bytes={size}, sha256={digest}"
        )
    return {"path": str(path), "bytes": size, "sha256": digest}


def _tile_slices(height: int, width: int) -> list[tuple[slice, slice]]:
    if height % GRID_SIZE or width % GRID_SIZE:
        raise ValueError("image dimensions must be divisible by the 4x4 tile grid")
    th, tw = height // GRID_SIZE, width // GRID_SIZE
    return [
        (slice(row * th, (row + 1) * th), slice(col * tw, (col + 1) * tw))
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
    ]


def _normalized_shift_similarity(
    current: np.ndarray, previous: np.ndarray, radius: int
) -> float:
    if current.shape != previous.shape or current.ndim != 2:
        raise ValueError("shift similarity expects equal two-dimensional arrays")
    best = -math.inf
    height, width = current.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cy = slice(max(0, dy), min(height, height + dy))
            py = slice(max(0, -dy), min(height, height - dy))
            cx = slice(max(0, dx), min(width, width + dx))
            px = slice(max(0, -dx), min(width, width - dx))
            value = float(np.sum(current[cy, cx] * previous[py, px], dtype=np.float64))
            best = max(best, value)
    return float(np.clip(best, -1.0, 1.0))


def _pixel_metrics(images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return blur, tangle, and normalized coherent-edge maps for [N,3,H,W]."""

    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(f"expected [N,3,H,W] RGB images, got {images.shape}")
    if not np.isfinite(images).all():
        raise ValueError("decoded image tensor contains non-finite values")
    rgb = images.astype(np.float64, copy=False)
    gray = 0.2989 * rgb[:, 0] + 0.5870 * rgb[:, 1] + 0.1140 * rgb[:, 2]
    tiles = _tile_slices(gray.shape[-2], gray.shape[-1])
    blur = np.empty(len(gray), dtype=np.float64)
    tangle = np.empty(len(gray), dtype=np.float64)
    coherent_edges = np.empty_like(gray)
    for index, source in enumerate(gray):
        smooth = ndimage.gaussian_filter(source, sigma=0.7, mode="reflect")
        gx = ndimage.sobel(smooth, axis=1, mode="reflect") / 8.0
        gy = ndimage.sobel(smooth, axis=0, mode="reflect") / 8.0
        magnitude = np.hypot(gx, gy)
        laplacian = ndimage.laplace(smooth, mode="reflect")
        variances = np.asarray(
            [float(np.var(source[ys, xs], dtype=np.float64)) for ys, xs in tiles]
        )
        active = np.argsort(-variances, kind="stable")[:ACTIVE_TILE_COUNT]
        q_values = np.asarray(
            [
                float(np.mean(laplacian[ys, xs] ** 2, dtype=np.float64))
                / (float(np.mean(magnitude[ys, xs] ** 2, dtype=np.float64)) + EPS)
                for ys, xs in tiles
            ]
        )
        blur[index] = -math.log(float(np.percentile(q_values[active], 25)) + EPS)

        jxx = ndimage.gaussian_filter(gx * gx, sigma=1.5, mode="reflect")
        jxy = ndimage.gaussian_filter(gx * gy, sigma=1.5, mode="reflect")
        jyy = ndimage.gaussian_filter(gy * gy, sigma=1.5, mode="reflect")
        trace = jxx + jyy
        coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (trace + EPS)
        coherence = np.clip(coherence, 0.0, 1.0)
        tile_tangle = np.asarray(
            [
                float(np.sum(trace[ys, xs] * (1.0 - coherence[ys, xs])))
                / (float(np.sum(trace[ys, xs])) + EPS)
                for ys, xs in tiles
            ]
        )
        tangle[index] = float(np.percentile(tile_tangle[active], 90))
        edge = magnitude * coherence
        norm = float(np.sqrt(np.sum(edge * edge, dtype=np.float64)))
        coherent_edges[index] = edge / (norm + EPS)
    return blur, tangle, coherent_edges


def _load_models(
    vae_snapshot: Path, resnet_weights: Path, device: str
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    # Heavy imports are delayed so --self-test remains CPU-only and artifact-free.
    import torch
    from diffusers.models import AutoencoderKL
    from torchvision.models import resnet18

    vae_identity = validate_vae_snapshot(vae_snapshot)
    resnet_identity = _validate_resnet_weights(resnet_weights)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if torch_device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    vae = AutoencoderKL.from_pretrained(
        str(vae_snapshot), local_files_only=True, torch_dtype=torch.float32
    ).to(device=torch_device, dtype=torch.float32).eval()
    network = resnet18(weights=None)
    try:
        state = torch.load(resnet_weights, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        state = torch.load(resnet_weights, map_location="cpu")
    network.load_state_dict(state, strict=True)
    network = network.to(device=torch_device, dtype=torch.float32).eval()
    for model in (vae, network):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return vae, network, vae_identity, resnet_identity


def _decode_pred_xstart(
    vae: Any,
    latents: np.ndarray,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if latents.ndim != 5 or latents.shape[2:] != (4, 32, 32):
        raise ValueError(f"expected pred_xstart [B,T,4,32,32], got {latents.shape}")
    flat = np.ascontiguousarray(latents.reshape(-1, 4, 32, 32), dtype=np.float32)
    images: list[np.ndarray] = []
    clipping: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(flat), batch_size):
            value = torch.from_numpy(flat[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            decoded = vae.decode(value / VAE_SCALING_FACTOR).sample
            normalized = (decoded + 1.0) / 2.0
            clip_fraction = ((normalized < 0.0) | (normalized > 1.0)).float().mean(
                dim=(1, 2, 3)
            )
            clipping.append(clip_fraction.cpu().numpy().astype(np.float64))
            images.append(normalized.clamp(0.0, 1.0).cpu().numpy().astype(np.float32))
    stacked = np.concatenate(images, axis=0)
    fractions = np.concatenate(clipping, axis=0)
    batch, checkpoints = latents.shape[:2]
    return (
        stacked.reshape(batch, checkpoints, 3, IMAGE_SIZE, IMAGE_SIZE),
        fractions.reshape(batch, checkpoints),
    )


def _resnet_forward(
    model: Any, images: np.ndarray, targets: np.ndarray, device: str, batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as torch_f

    if images.ndim != 4 or images.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected [N,3,256,256] images, got {images.shape}")
    if targets.shape != (len(images),):
        raise ValueError("target vector does not match images")
    means = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    embeddings: list[np.ndarray] = []
    log_odds: list[np.ndarray] = []
    cams: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            x = torch.from_numpy(images[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            )
            # Images are already 256x256, matching the V1 resize's short side.
            x = x[:, :, 16:240, 16:240]
            x = (x - means) / stds
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            layer4 = model.layer4(x)
            pooled = torch.flatten(model.avgpool(layer4), 1)
            logits = model.fc(pooled)
            target = torch.as_tensor(
                targets[start : start + len(logits)], device=device, dtype=torch.long
            )
            row = torch.arange(len(logits), device=device)
            target_logits = logits[row, target]
            masked = logits.clone()
            masked[row, target] = -torch.inf
            odds = target_logits - torch.logsumexp(masked, dim=1)
            weights = model.fc.weight[target]
            cam = torch.relu(torch.sum(layer4 * weights[:, :, None, None], dim=1))
            cam = cam / (torch.linalg.vector_norm(cam.flatten(1), dim=1)[:, None, None] + EPS)
            embeddings.append(pooled.cpu().numpy().astype(np.float64))
            log_odds.append(odds.cpu().numpy().astype(np.float64))
            cams.append(cam.cpu().numpy().astype(np.float64))
    return (
        np.concatenate(embeddings, axis=0),
        np.concatenate(log_odds, axis=0),
        np.concatenate(cams, axis=0),
    )


def _jump_track(fields: np.ndarray, radius: int) -> np.ndarray:
    if fields.ndim != 4:
        raise ValueError("expected shift fields [B,T,H,W]")
    output = np.empty((fields.shape[0], fields.shape[1] - 1), dtype=np.float64)
    for sample in range(fields.shape[0]):
        for step in range(1, fields.shape[1]):
            similarity = _normalized_shift_similarity(
                fields[sample, step], fields[sample, step - 1], radius
            )
            output[sample, step - 1] = 1.0 - similarity
    return output


def _extract_trace(
    record: TraceRecord,
    arrays: Mapping[str, np.ndarray],
    checkpoints: tuple[int, ...],
    vae: Any,
    resnet: Any,
    device: str,
    decode_batch_size: int,
    classifier_batch_size: int,
) -> Extraction:
    pred = np.ascontiguousarray(arrays["pred_xstart"][:, checkpoints], dtype=np.float32)
    images, clipping = _decode_pred_xstart(
        vae, pred, device=device, batch_size=decode_batch_size
    )
    batch, count = images.shape[:2]
    flat_images = images.reshape(batch * count, 3, IMAGE_SIZE, IMAGE_SIZE)
    blur, tangle, edges = _pixel_metrics(flat_images)
    flat_targets = np.repeat(np.asarray(record.classes, dtype=np.int64), count)
    embeddings, odds, cams = _resnet_forward(
        resnet,
        flat_images,
        flat_targets,
        device=device,
        batch_size=classifier_batch_size,
    )
    embeddings = embeddings.reshape(batch, count, -1)
    odds = odds.reshape(batch, count)
    cams = cams.reshape(batch, count, 7, 7)
    embedding_norm = np.linalg.norm(embeddings, axis=2)
    embedding_similarity = np.sum(embeddings[:, 1:] * embeddings[:, :-1], axis=2) / (
        embedding_norm[:, 1:] * embedding_norm[:, :-1] + EPS
    )
    tracks = {
        "decoded_local_blur_severity": blur.reshape(batch, count),
        "decoded_edge_tangle": tangle.reshape(batch, count),
        "decoded_coherent_edge_jump": _jump_track(
            edges.reshape(batch, count, IMAGE_SIZE, IMAGE_SIZE), EDGE_SHIFT_RADIUS
        ),
        "resnet18_embedding_cosine_jump": 1.0
        - np.clip(embedding_similarity, -1.0, 1.0),
        "resnet18_target_log_odds": odds,
        "resnet18_target_log_odds_drop": np.maximum(odds[:, :-1] - odds[:, 1:], 0.0),
        "resnet18_target_cam_jump": _jump_track(cams, CAM_SHIFT_RADIUS),
        "decoder_clipping_fraction": clipping,
    }
    if set(tracks) != set(LEVEL_TRACKS) | set(JUMP_TRACKS):
        raise AssertionError("visual track schema mismatch")
    for name, value in tracks.items():
        expected = (batch, count if name in LEVEL_TRACKS else count - 1)
        if value.shape != expected or not np.isfinite(value).all():
            raise RuntimeError(f"invalid extracted track {name}: {value.shape}")
    return Extraction(record=record, tracks=tracks)


def _scalar_reductions(name: str, values: np.ndarray) -> list[tuple[str, np.ndarray, str]]:
    reductions: list[tuple[str, np.ndarray, str]] = [
        (f"{name}__mean", np.mean(values, axis=1), "mean over selected checkpoints"),
        (f"{name}__maximum", np.max(values, axis=1), "maximum over selected checkpoints"),
        (f"{name}__last", values[:, -1], "value at the latest selected checkpoint"),
        (f"{name}__range", np.ptp(values, axis=1), "maximum minus minimum"),
    ]
    if name in LEVEL_TRACKS:
        reductions.append(
            (
                f"{name}__max_positive_jump",
                np.maximum(np.max(np.diff(values, axis=1), axis=1), 0.0),
                "maximum positive change between consecutive selected checkpoints",
            )
        )
    else:
        reductions.append(
            (f"{name}__sum", np.sum(values, axis=1), "sum over checkpoint transitions")
        )
    return reductions


def _combine(
    extractions: Sequence[Extraction], checkpoints: tuple[int, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    ordered = sorted(extractions, key=lambda item: item.record.global_seed)
    rows: list[dict[str, Any]] = []
    combined: dict[str, list[np.ndarray]] = {
        name: [] for name in (*LEVEL_TRACKS, *JUMP_TRACKS)
    }
    for run_index, extraction in enumerate(ordered):
        record = extraction.record
        for slot, class_id in enumerate(record.classes):
            rows.append(
                {
                    "sample_index": len(rows),
                    "run_index": run_index,
                    "global_seed": record.global_seed,
                    "class_slot": slot,
                    "class_id": class_id,
                    "trace_dir": str(record.root),
                    "endpoint_png_path": str(
                        record.root / f"images/{slot:02d}_class{class_id:04d}.png"
                    ),
                }
            )
        for name, value in extraction.tracks.items():
            combined[name].append(value)
    tracks = {
        name: np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float64)
        for name, parts in combined.items()
    }
    frame = pd.DataFrame(rows, columns=IDENTIFIER_COLUMNS)
    catalog_rows: list[dict[str, Any]] = []
    for name in (*LEVEL_TRACKS, *JUMP_TRACKS):
        availability = (
            "pre_innovation_current_model_output"
            if name in LEVEL_TRACKS
            else "pre_innovation_after_later_checkpoint_model_output"
        )
        family = (
            "decoded_pixels"
            if name.startswith("decoded_") or name.startswith("decoder_")
            else "fixed_resnet18"
        )
        for feature, values, reduction in _scalar_reductions(name, tracks[name]):
            frame[feature] = values
            catalog_rows.append(
                {
                    "feature": feature,
                    "track": name,
                    "family": family,
                    "availability": availability,
                    "latest_required_sampling_step": checkpoints[-1],
                    "latest_required_internal_timestep": STEPS - 1 - checkpoints[-1],
                    "observation_timing": "before_transition_at_latest_checkpoint",
                    "preterminal_actionable": checkpoints[-1] < STEPS - 1,
                    "track_length": len(checkpoints) if name in LEVEL_TRACKS else len(checkpoints) - 1,
                    "uses_realized_innovation": False,
                    "checkpoint_sampling_steps": ",".join(map(str, checkpoints)),
                    "checkpoint_internal_timesteps": ",".join(
                        str(STEPS - 1 - value) for value in checkpoints
                    ),
                    "reduction": reduction,
                    "formula_source": "protocol_snapshot.json",
                    "deployment_note": (
                        "fixed decoded-pred_xstart diagnostic; no endpoint, future "
                        "innovation, label, or quality posterior is used"
                    ),
                }
            )
    catalog = pd.DataFrame(catalog_rows)
    catalog.insert(0, "feature_index", np.arange(len(catalog), dtype=np.int32))
    features = catalog["feature"].tolist()
    if set(frame.columns[len(IDENTIFIER_COLUMNS) :]) != set(features):
        raise RuntimeError("sample feature table and catalog differ")
    if not np.isfinite(frame[features].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("sample feature table contains non-finite values")
    return frame, catalog, tracks


def _validate_output(root: Path) -> None:
    root = root.expanduser().absolute().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"invalid output directory: {root}")
    manifest = load_json(root / "manifest.json")
    completion = load_json(root / "completion.json")
    identity = dict(manifest)
    recorded_identity = identity.pop("identity_sha256", None)
    if recorded_identity != canonical_sha256(identity):
        raise RuntimeError("output manifest identity is invalid")
    if (
        completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != recorded_identity
        or completion.get("manifest_file_sha256") != sha256_file(root / "manifest.json")
        or completion.get("payload_sha256") != _canonical_self_hash(
            completion, "payload_sha256"
        )
    ):
        raise RuntimeError("output completion receipt is invalid")
    members = manifest.get("files")
    if not isinstance(members, list) or not all(isinstance(item, dict) for item in members):
        raise RuntimeError("output manifest files are malformed")
    expected = {item.get("name") for item in members} | {"manifest.json", "completion.json"}
    actual = {path.name for path in root.iterdir() if path.is_file() and not path.is_symlink()}
    if expected != actual:
        raise RuntimeError(f"output member set changed: {actual} != {expected}")
    for item in members:
        path = root / str(item["name"])
        _require_regular(path, "output payload")
        if path.stat().st_size != item.get("bytes") or sha256_file(path) != item.get("sha256"):
            raise RuntimeError(f"output payload changed: {path}")
    inventory = load_json(root / "source_inventory.json")
    with np.load(root / "time_series.npz", allow_pickle=False) as archive:
        records = inventory.get("time_series_arrays")
        if not isinstance(records, dict) or set(archive.files) != set(records):
            raise RuntimeError("time-series inventory/member set differs")
        for name in archive.files:
            value = np.ascontiguousarray(archive[name])
            if _array_record(value) != records[name] or not np.isfinite(value).all():
                raise RuntimeError(f"time-series array validation failed: {name}")
    frame = pd.read_csv(root / "sample_features.csv")
    catalog = pd.read_csv(root / "feature_catalog.csv")
    names = catalog["feature"].astype(str).tolist()
    if (
        tuple(frame.columns[: len(IDENTIFIER_COLUMNS)]) != IDENTIFIER_COLUMNS
        or set(frame.columns[len(IDENTIFIER_COLUMNS) :]) != set(names)
        or not np.isfinite(frame[names].to_numpy(dtype=np.float64)).all()
    ):
        raise RuntimeError("output scalar features/catalog validation failed")


def _validate_record_collection(
    records: Sequence[TraceRecord], args: argparse.Namespace
) -> None:
    if not records:
        raise RuntimeError("no validated trace records")
    seeds = [record.global_seed for record in records]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError(f"duplicate global seeds across traces: {seeds}")
    if len({record.classes for record in records}) != 1:
        raise RuntimeError("ordered class list differs across traces")
    if len({record.scientific_fingerprint_sha256 for record in records}) != 1:
        raise RuntimeError("scientific sampler fingerprint differs across traces")
    if args.expected_seeds is not None and tuple(sorted(seeds)) != tuple(
        sorted(args.expected_seeds)
    ):
        raise RuntimeError("observed trace seeds differ from --expected-seeds")
    if args.expected_classes is not None and records[0].classes != args.expected_classes:
        raise RuntimeError("observed class order differs from --expected-classes")


def publish(args: argparse.Namespace) -> Path:
    trace_dirs, source_inventories, inventory_bindings = _collect_trace_dirs(args)
    checkpoints = args.checkpoints
    # Validate cheap static model artifacts before the expensive trace pass.
    vae_identity = validate_vae_snapshot(args.vae_snapshot)
    resnet_identity = _validate_resnet_weights(args.resnet18_weights)
    if args.dry_run:
        records: list[TraceRecord] = []
        for index, path in enumerate(trace_dirs, start=1):
            print(f"validating trace {index}/{len(trace_dirs)}: {path}", flush=True)
            record, arrays = load_validated_trace(path)
            _check_inventory_trace_binding(record, inventory_bindings.get(path))
            records.append(record)
            del arrays
        _validate_record_collection(records, args)
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_VALID",
                    "trace_count": len(records),
                    "sample_count": sum(len(record.classes) for record in records),
                    "checkpoints": list(checkpoints),
                    "vae": vae_identity,
                    "resnet18": resnet_identity,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return args.output_dir

    vae, resnet, loaded_vae_identity, loaded_resnet_identity = _load_models(
        args.vae_snapshot, args.resnet18_weights, args.device
    )
    if loaded_vae_identity != vae_identity or loaded_resnet_identity != resnet_identity:
        raise RuntimeError("model artifacts changed between validation and loading")
    records = []
    extractions: list[Extraction] = []
    for index, path in enumerate(trace_dirs, start=1):
        print(f"validating trace {index}/{len(trace_dirs)}: {path}", flush=True)
        record, arrays = load_validated_trace(path)
        _check_inventory_trace_binding(record, inventory_bindings.get(path))
        records.append(record)
        print(f"decoding/extracting trace {index}/{len(trace_dirs)}: seed={record.global_seed}", flush=True)
        extractions.append(
            _extract_trace(
                record,
                arrays,
                checkpoints,
                vae,
                resnet,
                args.device,
                args.decode_batch_size,
                args.classifier_batch_size,
            )
        )
        del arrays
    _validate_record_collection(records, args)
    frame, catalog, tracks = _combine(extractions, checkpoints)
    ordered = sorted(records, key=lambda record: record.global_seed)
    time_series: dict[str, np.ndarray] = {
        "sample_index": frame["sample_index"].to_numpy(np.int32),
        "global_seed": frame["global_seed"].to_numpy(np.int64),
        "class_slot": frame["class_slot"].to_numpy(np.int16),
        "class_id": frame["class_id"].to_numpy(np.int16),
        "selected_sampling_step": np.asarray(checkpoints, dtype=np.int16),
        "selected_internal_timestep": np.asarray(
            [STEPS - 1 - value for value in checkpoints], dtype=np.int16
        ),
        "jump_from_sampling_step": np.asarray(checkpoints[:-1], dtype=np.int16),
        "jump_to_sampling_step": np.asarray(checkpoints[1:], dtype=np.int16),
        **tracks,
    }
    if not all(np.isfinite(value).all() for value in time_series.values()):
        raise RuntimeError("time-series payload contains non-finite values")

    output = args.output_dir.expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite existing output path: {output}")
    output = output.resolve()
    if any(root == output or root in output.parents or output in root.parents for root in trace_dirs):
        raise RuntimeError("output must not overlap a trace input")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        shutil.copyfile(Path(__file__).resolve(), staging / "analysis_source.py")
        atomic_json_dump(PROTOCOL, staging / "protocol_snapshot.json")
        frame.to_csv(staging / "sample_features.csv", index=False, float_format="%.17g")
        catalog.to_csv(staging / "feature_catalog.csv", index=False)
        np.savez_compressed(staging / "time_series.npz", **time_series)
        time_inventory = {
            name: _array_record(value) for name, value in sorted(time_series.items())
        }
        source_inventory = {
            "analysis_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "imported_validation_helper": {
                "path": str(ROOT / "experiments/analyze_dit_bad_good_custom_traces.py"),
                "sha256": sha256_file(
                    ROOT / "experiments/analyze_dit_bad_good_custom_traces.py"
                ),
                "contract": "load_validated_trace",
            },
            "input_label_free_source_inventories": source_inventories,
            "ordered_classes": list(ordered[0].classes),
            "ordered_seeds": [record.global_seed for record in ordered],
            "scientific_fingerprint_sha256": ordered[0].scientific_fingerprint_sha256,
            "trace_runs": [
                {**asdict(record), "root": str(record.root), "classes": list(record.classes)}
                for record in ordered
            ],
            "time_series_arrays": time_inventory,
        }
        atomic_json_dump(source_inventory, staging / "source_inventory.json")
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "vae": vae_identity,
            "resnet18": resnet_identity,
            "device": args.device,
            "dtype": "float32",
            "offline_only": True,
            "models_loaded_once": True,
            "decoded_images_saved": False,
            "selected_sampling_steps": list(checkpoints),
            "selected_internal_timesteps": [STEPS - 1 - value for value in checkpoints],
            "supervision_audit": PROTOCOL["supervision_policy"],
        }
        atomic_json_dump(provenance, staging / "provenance.json")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "COMPLETE_LABEL_FREE_VISUAL_TRACK_EXTRACTION",
            "trace_count": len(ordered),
            "sample_count": len(frame),
            "ordered_classes": list(ordered[0].classes),
            "ordered_seeds": [record.global_seed for record in ordered],
            "selected_sampling_steps": list(checkpoints),
            "selected_internal_timesteps": [STEPS - 1 - value for value in checkpoints],
            "level_track_count": len(LEVEL_TRACKS),
            "jump_track_count": len(JUMP_TRACKS),
            "scalar_feature_count": len(catalog),
            "time_series_array_count": len(time_series),
            "decoded_images_saved": False,
            "labels_read_or_emitted": False,
            "interpretation": (
                "diagnostic visual/semantic witnesses only; target log-odds is not a "
                "quality posterior and no feature is calibrated or selected here"
            ),
        }
        atomic_json_dump(summary, staging / "summary.json")
        payloads = []
        for path in sorted(staging.iterdir()):
            if path.name in {"manifest.json", "completion.json"}:
                continue
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"unexpected staging entry: {path}")
            payloads.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "complete",
            "analysis_source_sha256": sha256_file(staging / "analysis_source.py"),
            "protocol_snapshot_sha256": sha256_file(staging / "protocol_snapshot.json"),
            "source_inventory_sha256": sha256_file(staging / "source_inventory.json"),
            "provenance_sha256": sha256_file(staging / "provenance.json"),
            "files": payloads,
        }
        manifest["identity_sha256"] = canonical_sha256(manifest)
        atomic_json_dump(manifest, staging / "manifest.json")
        completion = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "summary_file_sha256": sha256_file(staging / "summary.json"),
        }
        completion["payload_sha256"] = _canonical_self_hash(completion, "payload_sha256")
        atomic_json_dump(completion, staging / "completion.json")
        _validate_output(staging)
        staging.rename(output)
        _validate_output(output)
        return output
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def self_test() -> None:
    rng = np.random.default_rng(20260827)
    height = width = 64
    checker = (np.indices((height, width)).sum(axis=0) % 2).astype(np.float32)
    sharp = np.stack([checker, checker, checker], axis=0)
    blurred_gray = ndimage.gaussian_filter(checker, sigma=3.0, mode="reflect")
    blurred = np.stack([blurred_gray, blurred_gray, blurred_gray], axis=0)
    noise = rng.uniform(0.0, 1.0, size=(3, height, width)).astype(np.float32)
    images = np.stack([sharp, blurred, noise], axis=0)
    blur, tangle, edges = _pixel_metrics(images)
    if not np.isfinite(blur).all() or not np.isfinite(tangle).all():
        raise AssertionError("synthetic pixel metrics are non-finite")
    if not blur[1] > blur[0]:
        raise AssertionError(f"blur witness ordering failed: {blur}")
    reference = np.zeros_like(edges[0])
    reference[12:50, 10:48] = rng.uniform(0.1, 1.0, size=(38, 38))
    reference /= np.linalg.norm(reference) + EPS
    shifted = np.zeros_like(reference)
    shifted[:, 3:] = reference[:, :-3]
    shifted /= np.linalg.norm(shifted) + EPS
    if _normalized_shift_similarity(shifted, reference, EDGE_SHIFT_RADIUS) < 0.999:
        raise AssertionError("shift-tolerant coherent-edge matching failed")
    synthetic_fields = np.stack([edges[:2], edges[:2]], axis=0)
    jumps = _jump_track(synthetic_fields, EDGE_SHIFT_RADIUS)
    if jumps.shape != (2, 1) or not np.isfinite(jumps).all():
        raise AssertionError("synthetic jump track failed")
    levels = np.asarray([[1.0, 2.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    reductions = _scalar_reductions("decoded_edge_tangle", levels)
    if len(reductions) != 5 or any(value.shape != (2,) for _, value, _ in reductions):
        raise AssertionError("scalar reductions failed")
    if DEFAULT_CHECKPOINTS != tuple(range(69, 150, 10)) or DEFAULT_INTERNAL_TIMESTEPS != tuple(
        range(180, 99, -10)
    ):
        raise AssertionError("default checkpoint mapping changed")
    print(
        "self-test passed: synthetic blur ordering, 4px shift-tolerant coherent "
        "edges, finite tangle/jump tracks, reductions, and k/t checkpoint mapping"
    )


def build_parser() -> argparse.ArgumentParser:
    vae_snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--stabilityai--sd-vae-ft-mse/snapshots"
        / VAE_REVISION
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-inventory",
        action="append",
        type=Path,
        help="Label-free primary/posterior source_inventory.json; repeatable.",
    )
    parser.add_argument("--trace-dir", action="append", type=Path)
    parser.add_argument("--trace-root", action="append", type=Path)
    parser.add_argument("--trace-glob", default="*")
    parser.add_argument("--expected-seeds", type=_parse_csv_ints)
    parser.add_argument("--expected-classes", type=_parse_csv_ints)
    parser.add_argument("--checkpoints", type=_parse_csv_ints, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--vae-snapshot", type=Path, default=vae_snapshot)
    parser.add_argument(
        "--resnet18-weights",
        type=Path,
        default=Path("/home/zhoushunyu/.cache/torch/hub/checkpoints")
        / RESNET18_FILENAME,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode-batch-size", type=int, default=27)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --self-test is used")
    if args.decode_batch_size <= 0 or args.classifier_batch_size <= 0:
        parser.error("batch sizes must be positive")
    checkpoints = tuple(args.checkpoints)
    if (
        len(checkpoints) < 2
        or tuple(sorted(checkpoints)) != checkpoints
        or checkpoints[0] < 0
        or checkpoints[-1] > 149
    ):
        parser.error("--checkpoints must be >=2 unique ascending k values within 0..149")
    args.checkpoints = checkpoints
    args.vae_snapshot = args.vae_snapshot.expanduser().absolute().resolve()
    args.resnet18_weights = args.resnet18_weights.expanduser().absolute().resolve()
    output = publish(args)
    if not args.dry_run:
        print(f"published immutable label-free visual tracks: {output}")


if __name__ == "__main__":
    main()
