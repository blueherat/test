#!/usr/bin/env python3
"""Extract sealed label-free endpoint embeddings from validated DiT traces.

The extractor is reusable for future disjoint pools: pass one or more
label-free ``source_inventory.json`` files, an exact expected seed set, and an
output directory.  It never opens a label/review/score file and never emits a
decoded image.  Endpoint PNG bytes are validated against their trace manifests
before two fixed representations are evaluated:

* torch-fidelity Inception-v3-compat final average-pool feature (2048-D), the
  standard representation used by the FID implementation in this environment;
* locally pinned facebook/dinov2-with-registers-large final normalized CLS
  token (1024-D).

The output contains raw embeddings and sample keys only.  No centroid,
distance, class reference, threshold, rank, AUC, or label is computed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


SCHEMA_VERSION = 1
EXPERIMENT = "dit_endpoint_embeddings_label_free_v1"
EXPECTED_CLASSES = (207, 602, 795)
REPRESENTATIONS = {
    "inception_fid_pool2048": 2048,
    "dinov2_registers_large_cls1024": 1024,
}
INCEPTION_WEIGHTS_SHA256 = (
    "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"
)
DINO_FILES_SHA256 = {
    "config.json": "03eee42f646659a9480f8911a81fdd81efeedd7ff39083c8e36398068daf72f5",
    "model.safetensors": "edccedab2c4e164e80833096de89a32a6e8d7365870499a066a61dbc8894b42b",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
}
DEFAULT_BASE = Path("/data/users/zhoushunyu/eqvae/cross_scale_evidence")
DEFAULT_SOURCE_INVENTORIES = (
    DEFAULT_BASE / "bad_good_metric_confirmation_v1/custom_label_free_v1/source_inventory.json",
    DEFAULT_BASE
    / "bad_good_metric_confirmation_expansion_v1/primary_label_free_v1/source_inventory.json",
)
DEFAULT_EXPECTED_SEEDS = tuple(range(50, 250))
DEFAULT_OUTPUT = (
    DEFAULT_BASE
    / "bad_good_metric_confirmation_expansion_v1/endpoint_embeddings_label_free_v1"
)
DEFAULT_INCEPTION_WEIGHTS = Path(
    "/home/zhoushunyu/.cache/torch/hub/checkpoints/"
    "pt_inception-2015-12-05-6726825d.pth"
)
DEFAULT_DINO_SNAPSHOT = Path(
    "/home/zhoushunyu/.cache/huggingface/hub/"
    "models--facebook--dinov2-with-registers-large/snapshots/"
    "e4c89a4e05589de9b3e188688a303d0f3c04d0f3"
)
FORBIDDEN_INPUT_TOKENS = (
    "consensus",
    "review",
    "adjudicat",
    "label_lock",
    "candidate_score",
    "calibrated_alert",
)
FORBIDDEN_OUTPUT_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}
PROTOCOL = {
    "schema_version": SCHEMA_VERSION,
    "experiment": EXPERIMENT,
    "status": "LABEL_FREE_ENDPOINT_REPRESENTATION_EXTRACTION_ONLY",
    "input": {
        "grain": "one terminal 256x256 RGB PNG per (global_seed,class_id)",
        "validation": (
            "source-analysis manifest/completion/source-inventory binding, trace "
            "manifest/completion identity, endpoint byte hash, mode, size, and pixel hash"
        ),
    },
    "representations": {
        "inception_fid_pool2048": {
            "model": "torch-fidelity Inception-v3-compat",
            "feature": "2048-D final adaptive-average-pool activation",
            "input": "original RGB uint8 PNG",
            "preprocessing": (
                "torch-fidelity internal TensorFlow-compatible bilinear resize to "
                "299x299 followed by (uint8-128)/128"
            ),
            "fid_semantics": (
                "this is the individual-image representation used by FID; no small-"
                "group FID is computed or treated as an individual score"
            ),
        },
        "dinov2_registers_large_cls1024": {
            "model": "facebook/dinov2-with-registers-large",
            "feature": "1024-D final normalized CLS token, last_hidden_state[:,0,:]",
            "input": "original RGB PNG through the pinned BitImageProcessor",
            "preprocessing": (
                "bicubic resize shortest edge 256, center crop 224, rescale 1/255, "
                "ImageNet mean/std normalization"
            ),
            "attention": "eager FP32 inference",
        },
    },
    "excluded_local_representation": {
        "google/siglip2-base-patch16-256": (
            "available locally but deliberately excluded to keep the preregistered "
            "family small; SigLIP2 is not silently substituted for CLIP"
        )
    },
    "supervision_policy": {
        "labels_read_or_emitted": False,
        "reviews_read": False,
        "candidate_scores_read": False,
        "distances_or_centroids_computed": False,
        "auc_or_selection_computed": False,
        "images_saved": False,
    },
    "timing": {
        "availability": "terminal endpoint only",
        "preterminal_actionable": False,
        "interpretation": "retrospective diagnostic/control, never an online trigger",
    },
}


@dataclass(frozen=True)
class EndpointRecord:
    global_seed: int
    class_slot: int
    class_id: int
    trace_root: Path
    trace_identity_sha256: str
    endpoint_path: Path
    endpoint_sha256: str
    endpoint_pixel_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_identity(value: Mapping[str, Any]) -> str:
    copied = dict(value)
    copied.pop("identity_sha256", None)
    return canonical_sha256(copied)


def canonical_self_hash(value: Mapping[str, Any], key: str) -> str:
    copied = dict(value)
    copied.pop(key, None)
    return canonical_sha256(copied)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_int_set(text: str) -> tuple[int, ...]:
    values: set[int] = set()
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" in token:
            pieces = token.split(":")
            if len(pieces) != 2:
                raise argparse.ArgumentTypeError(f"invalid half-open range: {token}")
            start, stop = map(int, pieces)
            if stop <= start:
                raise argparse.ArgumentTypeError(f"empty/reversed range: {token}")
            values.update(range(start, stop))
        else:
            values.add(int(token))
    if not values:
        raise argparse.ArgumentTypeError("integer set cannot be empty")
    return tuple(sorted(values))


def _require_regular(path: Path, description: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{description} must be a real regular file: {path}")


def _require_clean_input(path: Path, description: str) -> None:
    lowered = str(path).lower()
    if any(token in lowered for token in FORBIDDEN_INPUT_TOKENS):
        raise RuntimeError(f"{description} path looks supervised/forbidden: {path}")


def validate_manifest_members(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"manifest has no bound payload members: {root}")
    members: dict[str, Any] = {}
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise RuntimeError(f"malformed manifest member: {root}")
        name = item["name"]
        if name in members or Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError(f"unsafe/duplicate manifest member: {name}")
        path = root / name
        _require_regular(path, "manifest member")
        if path.stat().st_size != item.get("bytes") or sha256_file(path) != item.get(
            "sha256"
        ):
            raise RuntimeError(f"manifest member changed: {path}")
        members[name] = item
    return members


def validate_source_inventory(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate a label-free analysis envelope and return trace bindings only."""

    path = path.expanduser().absolute().resolve()
    _require_clean_input(path, "source inventory")
    _require_regular(path, "source inventory")
    if path.name != "source_inventory.json":
        raise RuntimeError("--source-inventory must name source_inventory.json")
    root = path.parent
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    summary_path = root / "summary.json"
    for candidate, description in (
        (manifest_path, "source manifest"),
        (completion_path, "source completion"),
        (summary_path, "source summary"),
    ):
        _require_regular(candidate, description)
    manifest = read_json(manifest_path)
    completion = read_json(completion_path)
    summary = read_json(summary_path)
    inventory = read_json(path)
    manifest_identity = canonical_identity(manifest)
    if (
        manifest.get("identity_sha256") != manifest_identity
        or manifest.get("status") != "complete"
        or manifest.get("source_inventory_sha256") != sha256_file(path)
        or completion.get("complete") is not True
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_identity_sha256") != manifest_identity
        or summary.get("labels_joined") is not False
        or summary.get("label_counts") != {"unlabeled": summary.get("sample_count")}
        or inventory.get("locked_consensus") is not None
    ):
        raise RuntimeError(f"source analysis is not sealed label-free: {root}")
    members = validate_manifest_members(root, manifest)
    if "source_inventory.json" not in members or "summary.json" not in members:
        raise RuntimeError(f"source analysis lacks required bound payload: {root}")
    runs = inventory.get("trace_runs")
    ordered_seeds = inventory.get("ordered_seeds")
    ordered_classes = inventory.get("ordered_classes")
    ordered_identities = manifest.get("trace_identity_sha256_ordered")
    if (
        not isinstance(runs, list)
        or not isinstance(ordered_seeds, list)
        or ordered_classes != list(EXPECTED_CLASSES)
        or not isinstance(ordered_identities, list)
        or len(runs) != len(ordered_seeds)
        or len(runs) != len(ordered_identities)
    ):
        raise RuntimeError(f"source trace inventory is malformed: {path}")
    records: list[dict[str, Any]] = []
    for seed, identity, run in zip(ordered_seeds, ordered_identities, runs, strict=True):
        if (
            type(seed) is not int
            or not isinstance(run, dict)
            or run.get("global_seed") != seed
            or run.get("classes") != list(EXPECTED_CLASSES)
            or run.get("identity_sha256") != identity
        ):
            raise RuntimeError(f"source trace ordering/binding changed: {path}")
        trace_root = Path(str(run.get("root", ""))).expanduser().absolute().resolve()
        _require_clean_input(trace_root, "trace root")
        for key in (
            "identity_sha256",
            "manifest_sha256",
            "completion_sha256",
            "trace_sha256",
            "scientific_fingerprint_sha256",
        ):
            value = run.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise RuntimeError(f"trace binding lacks {key}: {trace_root}")
        records.append({**run, "root": trace_root})
    return records, {
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_identity_sha256": manifest_identity,
        "completion_sha256": sha256_file(completion_path),
        "trace_run_count": len(records),
        "ordered_seed_min": min(ordered_seeds),
        "ordered_seed_max": max(ordered_seeds),
        "scientific_fingerprint_sha256": inventory.get(
            "scientific_fingerprint_sha256"
        ),
    }


def _pixel_sha256(image: Image.Image) -> str:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return sha256_array(array)


def validate_trace_endpoints(run: Mapping[str, Any]) -> list[EndpointRecord]:
    root = Path(run["root"])
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"trace root must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    _require_regular(manifest_path, "trace manifest")
    _require_regular(completion_path, "trace completion")
    if (
        sha256_file(manifest_path) != run["manifest_sha256"]
        or sha256_file(completion_path) != run["completion_sha256"]
    ):
        raise RuntimeError(f"trace envelope bytes changed: {root}")
    manifest = read_json(manifest_path)
    completion = read_json(completion_path)
    identity_payload = manifest.get("identity")
    if not isinstance(identity_payload, dict):
        raise RuntimeError(f"trace identity payload is missing: {root}")
    identity = canonical_sha256(identity_payload)
    trace_protocol = identity_payload.get("protocol", {})
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise RuntimeError(f"trace outputs missing: {root}")
    outputs_identity = canonical_sha256(outputs)
    expected_completion = {
        "schema": 1,
        "identity_sha256": identity,
        "manifest_sha256": run["manifest_sha256"],
        "outputs_sha256": outputs_identity,
        "output_count": len(outputs),
    }
    if (
        manifest.get("identity_sha256") != identity
        or identity != run["identity_sha256"]
        or manifest.get("status") != "complete"
        or manifest.get("outputs_sha256") != outputs_identity
        or completion != expected_completion
        or trace_protocol.get("global_torch_seed") != run["global_seed"]
        or trace_protocol.get("class_ids_ordered") != list(EXPECTED_CLASSES)
        or identity_payload.get("observation_only") is not True
        or identity_payload.get("selection") is not None
        or identity_payload.get("intervention") is not None
        or identity_payload.get("quality_score") is not None
    ):
        raise RuntimeError(f"trace scientific identity is invalid: {root}")
    by_relative = {
        item.get("relative_path"): item for item in outputs if isinstance(item, dict)
    }
    trace_item = by_relative.get("trace.npz")
    if not isinstance(trace_item, dict) or trace_item.get("sha256") != run["trace_sha256"]:
        raise RuntimeError(f"trace array binding differs: {root}")
    endpoints: list[EndpointRecord] = []
    for class_slot, class_id in enumerate(EXPECTED_CLASSES):
        relative = f"images/{class_slot:02d}_class{class_id:04d}.png"
        item = by_relative.get(relative)
        if not isinstance(item, dict):
            raise RuntimeError(f"trace lacks endpoint {relative}: {root}")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise RuntimeError(f"unsafe endpoint path: {path}")
        _require_regular(path, "endpoint PNG")
        if (
            path.stat().st_size != item.get("bytes")
            or sha256_file(path) != item.get("sha256")
            or item.get("mode") != "RGB"
            or item.get("size") != [256, 256]
        ):
            raise RuntimeError(f"endpoint bytes/metadata changed: {path}")
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (256, 256):
                raise RuntimeError(f"endpoint decode metadata changed: {path}")
            pixel_sha = _pixel_sha256(image)
        if pixel_sha != item.get("pixel_sha256"):
            raise RuntimeError(f"endpoint decoded pixels changed: {path}")
        endpoints.append(
            EndpointRecord(
                global_seed=int(run["global_seed"]),
                class_slot=class_slot,
                class_id=class_id,
                trace_root=root,
                trace_identity_sha256=identity,
                endpoint_path=path,
                endpoint_sha256=item["sha256"],
                endpoint_pixel_sha256=pixel_sha,
            )
        )
    return endpoints


def collect_endpoints(
    source_inventories: Sequence[Path],
    expected_seeds: tuple[int, ...],
) -> tuple[list[EndpointRecord], list[dict[str, Any]], str]:
    all_runs: dict[int, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for path in source_inventories:
        runs, binding = validate_source_inventory(path)
        bindings.append(binding)
        fingerprints.add(str(binding["scientific_fingerprint_sha256"]))
        for run in runs:
            seed = int(run["global_seed"])
            if seed in all_runs:
                raise RuntimeError(f"duplicate seed across source inventories: {seed}")
            all_runs[seed] = run
    if len(fingerprints) != 1:
        raise RuntimeError("source inventories use different sampler fingerprints")
    missing = set(expected_seeds) - set(all_runs)
    if missing:
        raise RuntimeError(f"expected seeds are missing: {sorted(missing)}")
    ordered: list[EndpointRecord] = []
    for seed in expected_seeds:
        ordered.extend(validate_trace_endpoints(all_runs[seed]))
    expected_axis = {
        (seed, class_id) for seed in expected_seeds for class_id in EXPECTED_CLASSES
    }
    observed_axis = {(item.global_seed, item.class_id) for item in ordered}
    if len(ordered) != len(expected_axis) or observed_axis != expected_axis:
        raise RuntimeError("endpoint axis is not exact")
    return ordered, bindings, next(iter(fingerprints))


def validate_model_files(
    inception_weights: Path, dino_snapshot: Path
) -> dict[str, Any]:
    inception_weights = inception_weights.expanduser().resolve()
    dino_snapshot = dino_snapshot.expanduser().resolve()
    _require_regular(inception_weights, "Inception weights")
    if sha256_file(inception_weights) != INCEPTION_WEIGHTS_SHA256:
        raise RuntimeError("Inception compatibility weights differ from the pin")
    if not dino_snapshot.is_dir() or dino_snapshot.is_symlink():
        raise RuntimeError(f"DINO snapshot must be a real directory: {dino_snapshot}")
    dino_files = []
    for name, expected_sha in DINO_FILES_SHA256.items():
        path = dino_snapshot / name
        if not path.is_file():
            raise RuntimeError(f"DINO {name} is missing: {path}")
        resolved = path.resolve()
        _require_regular(resolved, f"resolved DINO {name}")
        observed = sha256_file(resolved)
        if observed != expected_sha:
            raise RuntimeError(f"DINO {name} differs from the pin")
        dino_files.append(
            {
                "name": name,
                "path": str(path),
                "resolved_path": str(resolved),
                "bytes": resolved.stat().st_size,
                "sha256": observed,
            }
        )
    return {
        "inception": {
            "path": str(inception_weights),
            "bytes": inception_weights.stat().st_size,
            "sha256": INCEPTION_WEIGHTS_SHA256,
        },
        "dinov2": {
            "snapshot": str(dino_snapshot),
            "model_id": "facebook/dinov2-with-registers-large",
            "revision": "e4c89a4e05589de9b3e188688a303d0f3c04d0f3",
            "files": dino_files,
        },
    }


def extract_embeddings(
    endpoints: Sequence[EndpointRecord],
    inception_weights: Path,
    dino_snapshot: Path,
    device: str,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch_fidelity
    import transformers
    from torch_fidelity.feature_extractor_inceptionv3 import (
        FeatureExtractorInceptionV3,
    )
    from transformers import AutoImageProcessor, AutoModel

    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    target = torch.device(device)
    inception = FeatureExtractorInceptionV3(
        "inception-v3-compat",
        ["2048"],
        feature_extractor_weights_path=str(inception_weights),
        feature_extractor_internal_dtype="float32",
    ).to(target)
    processor = AutoImageProcessor.from_pretrained(
        str(dino_snapshot), local_files_only=True
    )
    dino = AutoModel.from_pretrained(
        str(dino_snapshot),
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(target)
    inception.eval()
    dino.eval()
    inception_rows: list[np.ndarray] = []
    dino_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(endpoints), batch_size):
            batch = endpoints[start : start + batch_size]
            pil_images: list[Image.Image] = []
            uint8_images: list[np.ndarray] = []
            for item in batch:
                with Image.open(item.endpoint_path) as image:
                    rgb = image.convert("RGB")
                    rgb.load()
                    pil_images.append(rgb.copy())
                    uint8_images.append(np.asarray(rgb, dtype=np.uint8))
            inception_input = torch.from_numpy(
                np.stack(uint8_images, axis=0).transpose(0, 3, 1, 2).copy()
            ).to(target)
            inception_value = inception(inception_input)[0]
            dino_input = processor(images=pil_images, return_tensors="pt")[
                "pixel_values"
            ].to(device=target, dtype=torch.float32)
            dino_output = dino(pixel_values=dino_input, return_dict=True)
            dino_value = dino_output.last_hidden_state[:, 0, :]
            inception_rows.append(inception_value.detach().cpu().numpy().astype(np.float32))
            dino_rows.append(dino_value.detach().cpu().numpy().astype(np.float32))
    arrays = {
        "inception_fid_pool2048": np.concatenate(inception_rows, axis=0),
        "dinov2_registers_large_cls1024": np.concatenate(dino_rows, axis=0),
    }
    for name, dimension in REPRESENTATIONS.items():
        value = arrays[name]
        if value.shape != (len(endpoints), dimension) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid extracted representation: {name} {value.shape}")
        if np.any(np.linalg.norm(value.astype(np.float64), axis=1) <= 0):
            raise RuntimeError(f"zero-norm representation: {name}")
    source_files = {}
    for name, obj in (
        ("torch_fidelity_inception", FeatureExtractorInceptionV3),
        ("transformers_dino_model", dino.__class__),
        ("transformers_dino_processor", processor.__class__),
    ):
        source = Path(str(inspect.getsourcefile(obj))).resolve()
        _require_regular(source, f"{name} implementation source")
        source_files[name] = {
            "path": str(source), "bytes": source.stat().st_size, "sha256": sha256_file(source)
        }
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": Image.__version__ if hasattr(Image, "__version__") else None,
        "torch": torch.__version__,
        "torch_fidelity": getattr(torch_fidelity, "__version__", None),
        "transformers": transformers.__version__,
        "device": str(target),
        "dtype": "float32",
        "deterministic_algorithms": True,
        "tf32": False,
        "models_loaded_once": True,
        "offline_only": True,
        "implementation_sources": source_files,
    }
    return arrays, runtime


def _array_record(value: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "raw_sha256": sha256_array(value),
    }


def _payload_record(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def publish(
    output: Path,
    endpoints: Sequence[EndpointRecord],
    embeddings: Mapping[str, np.ndarray],
    source_bindings: Sequence[Mapping[str, Any]],
    sampler_fingerprint: str,
    model_identity: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> None:
    output = output.expanduser().absolute()
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        shutil.copyfile(Path(__file__).resolve(), staging / "analysis_source.py")
        write_json(staging / "protocol_snapshot.json", PROTOCOL)
        with (staging / "sample_index.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "sample_index",
                    "global_seed",
                    "class_slot",
                    "class_id",
                    "trace_root",
                    "trace_identity_sha256",
                    "endpoint_png_path",
                    "endpoint_sha256",
                    "endpoint_pixel_sha256",
                ),
            )
            writer.writeheader()
            for index, item in enumerate(endpoints):
                writer.writerow(
                    {
                        "sample_index": index,
                        "global_seed": item.global_seed,
                        "class_slot": item.class_slot,
                        "class_id": item.class_id,
                        "trace_root": str(item.trace_root),
                        "trace_identity_sha256": item.trace_identity_sha256,
                        "endpoint_png_path": str(item.endpoint_path),
                        "endpoint_sha256": item.endpoint_sha256,
                        "endpoint_pixel_sha256": item.endpoint_pixel_sha256,
                    }
                )
        np.savez_compressed(staging / "embeddings.npz", **embeddings)
        catalog_rows = []
        for name, dimension in REPRESENTATIONS.items():
            catalog_rows.append(
                {
                    "representation": name,
                    "dimension": dimension,
                    "availability": "terminal_endpoint_only",
                    "preterminal_actionable": False,
                    "normalized_in_product": False,
                    "distance_or_score_in_product": False,
                    "formula_source": "protocol_snapshot.json",
                }
            )
        with (staging / "representation_catalog.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(catalog_rows[0]))
            writer.writeheader()
            writer.writerows(catalog_rows)
        source_inventory = {
            "input_label_free_source_analyses": list(source_bindings),
            "sampler_scientific_fingerprint_sha256": sampler_fingerprint,
            "ordered_seeds": sorted({item.global_seed for item in endpoints}),
            "ordered_classes": list(EXPECTED_CLASSES),
            "ordered_trace_identity_sha256": list(
                dict.fromkeys(item.trace_identity_sha256 for item in endpoints)
            ),
            "ordered_endpoint_sha256": [item.endpoint_sha256 for item in endpoints],
            "embedding_arrays": {
                name: _array_record(value) for name, value in sorted(embeddings.items())
            },
        }
        write_json(staging / "source_inventory.json", source_inventory)
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
            "models": model_identity,
            "runtime": dict(runtime),
            "supervision_audit": PROTOCOL["supervision_policy"],
        }
        write_json(staging / "provenance.json", provenance)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "COMPLETE_LABEL_FREE_ENDPOINT_EMBEDDINGS",
            "sample_count": len(endpoints),
            "seed_count": len({item.global_seed for item in endpoints}),
            "ordered_classes": list(EXPECTED_CLASSES),
            "representations": REPRESENTATIONS,
            "labels_read_or_emitted": False,
            "distances_or_scores_computed": False,
            "images_saved": False,
            "preterminal_actionable": False,
        }
        write_json(staging / "summary.json", summary)
        payloads = [_payload_record(path) for path in sorted(staging.iterdir())]
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": "complete",
            "analysis_source_sha256": sha256_file(staging / "analysis_source.py"),
            "protocol_snapshot_sha256": sha256_file(staging / "protocol_snapshot.json"),
            "source_inventory_sha256": sha256_file(staging / "source_inventory.json"),
            "provenance_sha256": sha256_file(staging / "provenance.json"),
            "files": payloads,
        }
        manifest["identity_sha256"] = canonical_identity(manifest)
        write_json(staging / "manifest.json", manifest)
        completion = {
            "complete": True,
            "manifest_identity_sha256": manifest["identity_sha256"],
            "manifest_file_sha256": sha256_file(staging / "manifest.json"),
            "summary_file_sha256": sha256_file(staging / "summary.json"),
        }
        completion["payload_sha256"] = canonical_self_hash(completion, "payload_sha256")
        write_json(staging / "completion.json", completion)
        validate_output(staging)
        os.rename(staging, output)
        validate_output(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_output(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"invalid embedding product directory: {root}")
    manifest = read_json(root / "manifest.json")
    completion = read_json(root / "completion.json")
    summary = read_json(root / "summary.json")
    inventory = read_json(root / "source_inventory.json")
    identity = canonical_identity(manifest)
    members = validate_manifest_members(root, manifest)
    entries = list(root.iterdir())
    if any(not path.is_file() or path.is_symlink() for path in entries):
        raise RuntimeError("embedding product contains a directory/symlink")
    actual = {path.name for path in entries}
    expected = set(members) | {"manifest.json", "completion.json"}
    if actual != expected or any(
        path.suffix.lower() in FORBIDDEN_OUTPUT_SUFFIXES for path in root.iterdir()
    ):
        raise RuntimeError("embedding product file set/image policy changed")
    if (
        manifest.get("identity_sha256") != identity
        or manifest.get("status") != "complete"
        or manifest.get("experiment") != EXPERIMENT
        or completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != identity
        or completion.get("manifest_file_sha256") != sha256_file(root / "manifest.json")
        or completion.get("summary_file_sha256") != sha256_file(root / "summary.json")
        or completion.get("payload_sha256")
        != canonical_self_hash(completion, "payload_sha256")
        or summary.get("labels_read_or_emitted") is not False
        or summary.get("distances_or_scores_computed") is not False
        or summary.get("images_saved") is not False
        or summary.get("preterminal_actionable") is not False
    ):
        raise RuntimeError("embedding product envelope/supervision policy is invalid")
    with np.load(root / "embeddings.npz", allow_pickle=False) as archive:
        records = inventory.get("embedding_arrays")
        if not isinstance(records, dict) or set(archive.files) != set(REPRESENTATIONS):
            raise RuntimeError("embedding array member set changed")
        for name, dimension in REPRESENTATIONS.items():
            value = archive[name]
            if (
                value.shape != (summary.get("sample_count"), dimension)
                or not np.isfinite(value).all()
                or _array_record(value) != records.get(name)
            ):
                raise RuntimeError(f"embedding array validation failed: {name}")
    with (root / "sample_index.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != summary.get("sample_count") or any(
        key in rows[0] for key in ("label", "primary_label", "raw_consensus_label")
    ):
        raise RuntimeError("embedding sample index is invalid/supervised")
    keys = {(int(row["global_seed"]), int(row["class_id"])) for row in rows}
    if len(keys) != len(rows):
        raise RuntimeError("embedding sample keys are not unique")
    return {
        "manifest_identity_sha256": identity,
        "sample_count": len(rows),
        "seed_count": len({int(row["global_seed"]) for row in rows}),
        "representation_count": len(REPRESENTATIONS),
    }


def self_test() -> None:
    assert parse_int_set("1,3:6,2") == (1, 2, 3, 4, 5)
    rng = np.random.default_rng(9)
    temporary_parent = Path(tempfile.mkdtemp(prefix="endpoint-embedding-selftest-"))
    try:
        image_path = temporary_parent / "endpoint.png"
        image = Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8), "RGB")
        image.save(image_path)
        endpoints = [
            EndpointRecord(
                global_seed=1,
                class_slot=0,
                class_id=207,
                trace_root=temporary_parent,
                trace_identity_sha256="a" * 64,
                endpoint_path=image_path,
                endpoint_sha256=sha256_file(image_path),
                endpoint_pixel_sha256=_pixel_sha256(image),
            )
        ]
        embeddings = {
            name: rng.normal(size=(1, dimension)).astype(np.float32)
            for name, dimension in REPRESENTATIONS.items()
        }
        output = temporary_parent / "product"
        publish(
            output,
            endpoints,
            embeddings,
            source_bindings=[],
            sampler_fingerprint="b" * 64,
            model_identity={"synthetic": True},
            runtime={"synthetic": True},
        )
        receipt = validate_output(output)
        assert receipt["sample_count"] == 1
        try:
            publish(
                output,
                endpoints,
                embeddings,
                [],
                "b" * 64,
                {"synthetic": True},
                {"synthetic": True},
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("publisher overwrote an existing product")
    finally:
        shutil.rmtree(temporary_parent)
    print("synthetic label-free endpoint embedding self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-inventory",
        action="append",
        type=Path,
        help="Label-free source_inventory.json; repeatable.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=parse_int_set,
        default=DEFAULT_EXPECTED_SEEDS,
        help="Comma-separated integers and/or half-open ranges such as 50:250.",
    )
    parser.add_argument(
        "--expected-classes", type=parse_int_set, default=EXPECTED_CLASSES
    )
    parser.add_argument("--inception-weights", type=Path, default=DEFAULT_INCEPTION_WEIGHTS)
    parser.add_argument("--dino-snapshot", type=Path, default=DEFAULT_DINO_SNAPSHOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        self_test()
        return
    if args.validate_output is not None:
        print(json.dumps(validate_output(args.validate_output), sort_keys=True))
        return
    if tuple(args.expected_classes) != EXPECTED_CLASSES:
        raise ValueError(f"this extractor is pinned to classes {EXPECTED_CLASSES}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    inventories = tuple(args.source_inventory or DEFAULT_SOURCE_INVENTORIES)
    model_identity = validate_model_files(args.inception_weights, args.dino_snapshot)
    endpoints, source_bindings, fingerprint = collect_endpoints(
        inventories, tuple(args.expected_seeds)
    )
    print(
        f"validated label-free endpoints: {len(endpoints)} samples, "
        f"{len(args.expected_seeds)} seeds, classes={EXPECTED_CLASSES}"
    )
    if args.dry_run:
        print("dry-run complete: no model loaded and no output written")
        return
    arrays, runtime = extract_embeddings(
        endpoints,
        args.inception_weights.expanduser().resolve(),
        args.dino_snapshot.expanduser().resolve(),
        args.device,
        args.batch_size,
    )
    publish(
        args.output_dir,
        endpoints,
        arrays,
        source_bindings,
        fingerprint,
        model_identity,
        runtime,
    )
    print(f"published sealed label-free embedding product: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
