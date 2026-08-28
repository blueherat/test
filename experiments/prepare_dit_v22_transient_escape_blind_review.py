#!/usr/bin/env python3
"""Prepare the sealed blind review for the prospective DiT-v2.2 suffix test.

This builder is deliberately phase separated.  It authenticates the frozen
V1.2 lock, all 128 execution receipts, and the byte-sealed internal-selector
product before it is permitted to inspect or copy any prospective endpoint
PNG.  Public delivery contains only opaque images, sheets, rubrics, and empty
review templates.  Every lineage-bearing mapping, internal method assignment,
source hash, and qualification gold label is installed in a physically
separate private tree.

No FID, embedding, visual label, review response, B/E/O value, or selector
magnitude is an input.  The already sealed selector *attempt identities* are
opened only after the internal product has passed byte-for-byte validation;
they never affect public IDs, item order, sheet order, or pair side.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

sys.dont_write_bytecode = True

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "experiments/locks/dit_v22_transient_escape_prospective_lock_v1_2"
DEFAULT_INTERNAL_PRODUCT = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_v22_transient_escape_internal_v1"
)
DEFAULT_ANCHOR_PACK = Path(
    "/data/users/zhoushunyu/eqvae/cross_scale_evidence/"
    "dit_imagenet256_visual_anchor_review/"
    "class207_visual_anchor_pack_v1"
)

LOCK_KIND = "DIT_V22_TRANSIENT_ESCAPE_PROSPECTIVE_LOCK_V1_2"
LOCK_ID = "cd8154479f5f6f883ae21d6657a61ec91ff6d2c77f569e18ea589d83517671a9"
PROTOCOL_ID = "54b11c1ebb6e310c73bb14e27c18e0f1810b5598212e2dc0c9be915f861155c1"
PRODUCT_KIND = "DIT_V22_TRANSIENT_ESCAPE_INTERNAL_PRODUCT_V1"
RUNNER_NAME = "intervene_dit_v22_transient_escape_suffix"
RNG_NAMESPACE = "eqvae-dit-v22-h10-max-nonconformity-prospective-v1"

ANCHOR_PACK_NAME = "dit_imagenet256_class207_visual_anchor_pack_v1"
ANCHOR_MANIFEST_ID = "ca259d0a66762c85b01c3d341041a2cd41ee8a08f0dd3b9d849e4ab59d0f233a"
ANCHOR_COMPLETION_ID = "15c60954ed3f3f93926e64a99de53bb3444f36bac4bc6f9dedc4dbdeb4f32a21"

PUBLIC_KIND = "DIT_V22_TRANSIENT_ESCAPE_BLIND_DELIVERY_V1"
PRIVATE_KIND = "DIT_V22_TRANSIENT_ESCAPE_PRIVATE_MAPPING_V1"
NAMESPACE = "eqvae.dit.v22.transient-escape.blind-review.v1"
REVIEWER_COUNT = 3
JOB_COUNT = 128
BRANCH_COUNT = 5
FRESH_ATTEMPTS = (1, 2, 3, 4)
ABSOLUTE_COUNT = JOB_COUNT * BRANCH_COUNT
PAIR_COUNT = JOB_COUNT * len(FRESH_ATTEMPTS)
QUALIFICATION_COUNT = 7
IMAGE_SIZE = (256, 256)
IMAGE_MODE = "RGB"

ABSOLUTE_COLUMNS = (
    "image_id",
    "severity",
    "blur_fusion",
    "topology_misalignment",
    "valid",
    "localized_reason",
)
PRESERVATION_COLUMNS = (
    "pair_id",
    "preserved",
    "preferred_side",
    "valid",
    "reason",
)
QUALIFICATION_COLUMNS = (
    "qualification_id",
    "severity",
    "valid",
    "reason",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def self_hash(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
    return canonical_sha256(payload)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_any(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"expected a regular JSON file: {path}")
    try:
        return json.loads(
            path.read_bytes().decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid JSON file: {path}") from exc


def load_json(path: Path) -> dict[str, Any]:
    value = load_json_any(path)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def load_self_hashed(path: Path, key: str) -> dict[str, Any]:
    value = load_json(path)
    observed = value.get(key)
    if not isinstance(observed, str) or observed != self_hash(value, key):
        raise RuntimeError(f"self hash failed: {path}")
    return value


def write_bytes(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError(f"short write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    write_bytes(path, payload.encode("utf-8"))


def hidden_key(domain: str, *parts: Any) -> str:
    payload = "\0".join((NAMESPACE, domain, *map(str, parts))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe_relative_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str):
        raise RuntimeError("relative path is not a string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError(f"unsafe relative path: {value!r}")
    return path


def path_within(root: Path, relative: Any) -> Path:
    path = (root / safe_relative_path(relative)).resolve()
    if root != path and root not in path.parents:
        raise RuntimeError(f"path escaped root: {relative!r}")
    return path


def resolve_input_root(path: Path, label: str) -> Path:
    lexical = path.expanduser().absolute()
    if lexical.is_symlink():
        raise RuntimeError(f"{label} root must not be a symlink: {lexical}")
    return lexical.resolve()


def exact_file_records(root: Path, *, excluded: set[str]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"symlink in sealed tree: {path}")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded:
            records.append(
                {
                    "name": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def validate_exact_directories(root: Path, relative_files: Iterable[str]) -> None:
    expected: set[str] = set()
    for name in relative_files:
        parent = PurePosixPath(name).parent
        while parent != PurePosixPath("."):
            expected.add(parent.as_posix())
            parent = parent.parent
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()
    }
    if actual != expected:
        raise RuntimeError(f"sealed directory tree changed: {root}")


def validate_lock(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError(f"invalid V1.2 lock tree: {root}")
    manifest = load_self_hashed(root / "manifest.json", "identity_sha256")
    if (
        manifest.get("artifact_kind") != LOCK_KIND
        or manifest.get("status") != "complete"
        or manifest.get("identity_sha256") != LOCK_ID
    ):
        raise RuntimeError("wrong prospective V1.2 lock")
    records: dict[str, dict[str, Any]] = {}
    for row in manifest.get("files", []):
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise RuntimeError("invalid lock member record")
        name = str(row["name"])
        if name in records:
            raise RuntimeError("duplicate lock member record")
        records[name] = row
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() != "manifest.json"
    }
    if set(records) != actual:
        raise RuntimeError("prospective lock exact tree changed")
    validate_exact_directories(root, {*records, "manifest.json"})
    for name, row in records.items():
        path = root / name
        expected = {"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if row != expected:
            raise RuntimeError(f"prospective lock member changed: {name}")
    protocol = load_self_hashed(root / "protocol.json", "identity_sha256")
    config = load_json(root / "frozen_config.json")
    if (
        protocol.get("identity_sha256") != PROTOCOL_ID
        or protocol.get("identity_sha256") != manifest.get("protocol_identity_sha256")
        or protocol.get("status") != "EXECUTION_READY_UNOBSERVED_PROSPECTIVE_SUFFIXES"
        or len(protocol.get("jobs", [])) != JOB_COUNT
        or config.get("experiment") != "dit_v22_transient_escape_prospective_v1"
    ):
        raise RuntimeError("prospective protocol/config scope changed")
    return manifest, protocol, config


def validate_receipts(
    protocol: Mapping[str, Any], lock_identity: str
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    raw_root = Path(protocol["outputs"]["receipt_root"]).expanduser().absolute()
    if raw_root.is_symlink():
        raise RuntimeError("V1.2 receipt root must not be a symlink")
    root = raw_root.resolve()
    if not root.is_dir():
        raise RuntimeError("all 128 V1.2 receipts are required before image access")
    by_job: dict[int, dict[str, Any]] = {}
    receipt_seals: list[dict[str, Any]] = []
    seen_shards: set[int] = set()
    shard_count: int | None = None
    if any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("symlink in V1.2 receipt tree")
    directories = sorted(root.glob("shard_*_of_*"))
    if not directories:
        raise RuntimeError("no V1.2 shard receipts found")
    if set(root.iterdir()) != set(directories):
        raise RuntimeError("V1.2 receipt-root exact tree changed")
    for directory in directories:
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(f"invalid receipt directory: {directory}")
        actual_files = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        }
        if actual_files != {"receipt.json"}:
            raise RuntimeError(f"receipt directory exact tree changed: {directory}")
        validate_exact_directories(directory, {"receipt.json"})
        receipt = load_self_hashed(directory / "receipt.json", "identity_sha256")
        if (
            receipt.get("artifact_kind")
            != "DIT_V22_TRANSIENT_ESCAPE_PROSPECTIVE_SHARD_RECEIPT_V1"
            or receipt.get("status") != "complete"
            or receipt.get("execution_lock_identity_sha256") != lock_identity
            or receipt.get("protocol_identity_sha256") != protocol["identity_sha256"]
            or receipt.get("png_label_quality_B_E_O_FID_embedding_or_attempt_selection_used")
            is not False
        ):
            raise RuntimeError(f"receipt scope changed: {directory}")
        current_count = int(receipt["shard_count"])
        current_index = int(receipt["shard_index"])
        if shard_count is None:
            shard_count = current_count
        if (
            current_count != shard_count
            or current_index in seen_shards
            or not 0 <= current_index < current_count
        ):
            raise RuntimeError("receipt shard axis changed")
        seen_shards.add(current_index)
        indices = receipt.get("job_indices")
        outputs = receipt.get("outputs")
        if not isinstance(indices, list) or not isinstance(outputs, list):
            raise RuntimeError("receipt job/output lists are absent")
        if (
            any(not isinstance(row, dict) or "job_index" not in row for row in outputs)
            or len(indices) != len(set(map(int, indices)))
        ):
            raise RuntimeError("receipt contains an invalid or duplicate job row")
        output_by_job = {int(row["job_index"]): row for row in outputs}
        if len(output_by_job) != len(outputs):
            raise RuntimeError("receipt contains duplicate output records")
        expected_indices = {
            int(job["job_index"])
            for job in protocol["jobs"]
            if int(job["job_index"]) % current_count == current_index
        }
        if set(map(int, indices)) != set(output_by_job) or set(output_by_job) != expected_indices:
            raise RuntimeError("receipt job/output axis changed")
        expected_directory_name = f"shard_{current_index:02d}_of_{current_count:02d}"
        if directory.name != expected_directory_name:
            raise RuntimeError("receipt directory/shard binding changed")
        receipt_seals.append(
            {
                "relative_directory": directory.name,
                "shard_index": current_index,
                "shard_count": current_count,
                "identity_sha256": receipt["identity_sha256"],
                "receipt_file_sha256": sha256_file(directory / "receipt.json"),
                "job_indices": list(map(int, indices)),
            }
        )
        for job_index, record in output_by_job.items():
            if job_index in by_job:
                raise RuntimeError(f"duplicate receipt for job {job_index}")
            by_job[job_index] = dict(record)
    if shard_count is None or seen_shards != set(range(shard_count)) or set(by_job) != set(range(JOB_COUNT)):
        raise RuntimeError("receipts do not cover the exact 128-job axis")
    return by_job, sorted(receipt_seals, key=lambda row: row["shard_index"])


def validate_internal_product(
    root: Path,
    *,
    lock: Path,
    lock_identity: str,
    protocol_identity: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("sealed internal product is absent or has a symlink")
    manifest_path = root / "manifest.json"
    completion_path = root / "completion.json"
    manifest = load_self_hashed(manifest_path, "identity_sha256")
    completion = load_self_hashed(completion_path, "identity_sha256")
    expected_payload = {
        "distance_matrices.json",
        "extractor_source.py",
        "features.csv",
        "features.json",
        "frozen_config.json",
        "input_inventory.json",
        "sealed_selections.json",
    }
    records: dict[str, dict[str, Any]] = {}
    for row in manifest.get("files", []):
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise RuntimeError("invalid internal-product member record")
        name = str(row["name"])
        if name in records:
            raise RuntimeError("duplicate internal-product member record")
        records[name] = row
    actual_payload = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in {"manifest.json", "completion.json"}
    }
    if set(records) != expected_payload or actual_payload != expected_payload:
        raise RuntimeError("internal product exact tree changed")
    validate_exact_directories(
        root, {*expected_payload, "manifest.json", "completion.json"}
    )
    for name, row in records.items():
        path = root / name
        expected = {"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if row != expected:
            raise RuntimeError(f"internal product member changed: {name}")
    if (
        manifest.get("artifact_kind") != PRODUCT_KIND
        or manifest.get("status") != "complete"
        or manifest.get("scientific_role")
        != "prospective_sampler_internal_selection_sealed_before_external_judging"
        or manifest.get("lock_identity_sha256") != lock_identity
        or manifest.get("protocol_identity_sha256") != protocol_identity
        or manifest.get("counts")
        != {"jobs": JOB_COUNT, "feature_rows": JOB_COUNT * 3, "selection_rows": JOB_COUNT}
        or manifest.get("selector") != "step149_h10_argmax_fresh_mean_nonconformity"
        or manifest.get("attempt0_O_lowO_B_E_or_external_metric_computed") is not False
        or manifest.get("png_pixels_opened") is not False
        or manifest.get("all_outputs_retained") is not True
        or sha256_file(root / "extractor_source.py")
        != sha256_file(lock / "sources/extract_dit_v22_transient_escape_internal.py")
        or sha256_file(root / "frozen_config.json") != sha256_file(lock / "frozen_config.json")
    ):
        raise RuntimeError("internal product identity/scope changed")
    selection_path = root / "sealed_selections.json"
    selection_sha = sha256_file(selection_path)
    if (
        completion.get("complete") is not True
        or completion.get("product_identity_sha256") != manifest["identity_sha256"]
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("sealed_selections_file_sha256") != selection_sha
        or completion.get("external_judging_may_begin_after_this_product") is not True
    ):
        raise RuntimeError("internal product completion seal changed")
    inventory = load_self_hashed(root / "input_inventory.json", "identity_sha256")
    if (
        inventory.get("png_pixels_review_labels_B_E_O_quality_FID_or_embeddings_opened")
        is not False
        or inventory.get("trace_arrays_read") != ["internal_timestep", "target_pred_xstart"]
    ):
        raise RuntimeError("internal product input firewall changed")
    raw_selections = load_json_any(selection_path)
    if not isinstance(raw_selections, list) or len(raw_selections) != JOB_COUNT:
        raise RuntimeError("sealed selection row count changed")
    selections = [dict(row) for row in raw_selections if isinstance(row, dict)]
    if len(selections) != JOB_COUNT:
        raise RuntimeError("sealed selection has a non-object row")
    return manifest, selections, selection_sha


def validate_selection_rows(
    jobs: Sequence[Mapping[str, Any]], selections: Sequence[Mapping[str, Any]]
) -> dict[int, dict[str, Any]]:
    by_job: dict[int, dict[str, Any]] = {}
    jobs_by_index = {int(job["job_index"]): job for job in jobs}
    for raw in selections:
        row = dict(raw)
        job_index = int(row.get("job_index", -1))
        if job_index in by_job or job_index not in jobs_by_index:
            raise RuntimeError("sealed selection job identity changed")
        job = jobs_by_index[job_index]
        maximum = int(row.get("max_attempt", -1))
        medoid = int(row.get("medoid_attempt", -1))
        random_attempt = int(row.get("hash_random_attempt", -1))
        slot_map = {int(key): value for key, value in job["physical_attempt_to_anonymous_slot"].items()}
        if (
            row.get("global_seed") != job["global_seed"]
            or row.get("class_id") != job["class_id"]
            or row.get("rollback_step") != 149
            or row.get("primary_horizon") != 10
            or row.get("selector") != "argmax_h10_fresh_mean_nonconformity"
            or maximum not in FRESH_ATTEMPTS
            or medoid not in FRESH_ATTEMPTS
            or random_attempt != job["hash_random_control_attempt"]
            or row.get("max_slot") != slot_map[maximum]
            or row.get("medoid_slot") != slot_map[medoid]
            or row.get("hash_random_slot") != job["hash_random_control_slot"]
            or row.get("all_four_fresh_endpoints_must_remain_available_to_external_judge")
            is not True
        ):
            raise RuntimeError(f"sealed selection scope changed for job {job_index}")
        by_job[job_index] = row
    if set(by_job) != set(range(JOB_COUNT)):
        raise RuntimeError("sealed selections do not cover 128 jobs")
    return by_job


def validate_anchor_pack(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("fixed qualification anchor pack is invalid")
    manifest = load_self_hashed(root / "manifest.json", "identity_sha256")
    completion = load_self_hashed(root / "completion.json", "payload_sha256")
    if (
        manifest.get("identity_sha256") != ANCHOR_MANIFEST_ID
        or manifest.get("pack_name") != ANCHOR_PACK_NAME
        or manifest.get("role") != "REVIEWER_ONLY_FIXED_EXTERNAL_VISUAL_CALIBRATION"
        or manifest.get("frozen_before_current_pool") is not True
        or manifest.get("anchor_count") != QUALIFICATION_COUNT
        or completion.get("payload_sha256") != ANCHOR_COMPLETION_ID
        or completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != manifest["identity_sha256"]
        or completion.get("manifest_file_sha256") != sha256_file(root / "manifest.json")
        or completion.get("anchor_count") != QUALIFICATION_COUNT
    ):
        raise RuntimeError("fixed anchor pack identity/completion changed")
    payload_rows = manifest.get("payload_files")
    if not isinstance(payload_rows, list):
        raise RuntimeError("anchor payload records are absent")
    expected_files = {"manifest.json", "completion.json"}
    for row in payload_rows:
        if not isinstance(row, dict):
            raise RuntimeError("invalid anchor payload record")
        path = path_within(root, row.get("relative_path"))
        relative = path.relative_to(root).as_posix()
        expected_files.add(relative)
        if (
            path.stat().st_size != row.get("bytes")
            or sha256_file(path) != row.get("file_sha256")
        ):
            raise RuntimeError(f"anchor payload changed: {relative}")
    actual_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual_files != expected_files:
        raise RuntimeError("anchor pack exact tree changed")
    validate_exact_directories(root, expected_files)
    if (
        manifest.get("payload_file_count") != len(payload_rows)
        or manifest.get("pack_payload_sha256") != canonical_sha256(payload_rows)
        or completion.get("pack_payload_sha256") != manifest["pack_payload_sha256"]
        or completion.get("file_count") != len(expected_files)
    ):
        raise RuntimeError("anchor payload/completion binding changed")
    anchors = manifest.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != QUALIFICATION_COUNT:
        raise RuntimeError("anchor records changed")
    categories = [row.get("category") for row in anchors if isinstance(row, dict)]
    if categories.count("ordinary") != 5 or sum(str(value).startswith("clear_bad") for value in categories) != 2:
        raise RuntimeError("anchor gold categories changed")
    return manifest, [dict(row) for row in anchors]


def validate_png(path: Path, record: Mapping[str, Any]) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing source PNG: {path}")
    file_hash = sha256_file(path)
    expected_hash = record.get("sha256", record.get("file_sha256"))
    expected_bytes = record.get("bytes")
    if file_hash != expected_hash or path.stat().st_size != expected_bytes:
        raise RuntimeError(f"source PNG identity changed: {path}")
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != IMAGE_MODE or image.size != IMAGE_SIZE:
            raise RuntimeError(f"source PNG format changed: {path}")
        pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
    expected_pixel = record.get("pixel_sha256", record.get("pixel_rgb_sha256"))
    if expected_pixel is not None and pixel_hash != expected_pixel:
        raise RuntimeError(f"source PNG pixels changed: {path}")
    return file_hash, pixel_hash


def validate_output_metadata(
    job: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    lock_identity: str,
    protocol_identity: str,
    runner_sha: str,
) -> dict[int, dict[str, Any]]:
    raw_root = Path(job["outdir"]).expanduser().absolute()
    if raw_root.is_symlink():
        raise RuntimeError(f"suffix bundle root must not be a symlink: {raw_root}")
    root = raw_root.resolve()
    if not root.is_dir() or any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError(f"invalid completed suffix bundle: {root}")
    manifest_path = root / "manifest.json"
    results_path = root / "results.json"
    completion_path = root / "completion.json"
    manifest = load_self_hashed(manifest_path, "identity_sha256")
    results = load_self_hashed(results_path, "payload_sha256")
    completion = load_self_hashed(completion_path, "payload_sha256")
    binding = manifest.get("prospective_binding", {})
    input_trace = manifest.get("input_trace", {})
    target = manifest.get("target", {})
    rollback = manifest.get("rollback", {})
    streams = [row.get("seed") for row in manifest.get("branches", {}).get("fresh_stream_seeds", [])]
    if (
        manifest.get("runner") != RUNNER_NAME
        or manifest.get("posthoc_exploratory") is not False
        or manifest.get("method_claim_eligible") is not True
        or manifest.get("attempt_ranking_or_selection") is not False
        or manifest.get("quality_scores_or_labels_used_by_runner") is not False
        or manifest.get("FID_Inception_DINO_CLIP_or_embeddings_used") is not False
        or manifest.get("rng", {}).get("namespace") != RNG_NAMESPACE
        or manifest.get("runner_source", {}).get("sha256") != runner_sha
        or binding.get("lock_identity_sha256") != lock_identity
        or binding.get("protocol_identity_sha256") != protocol_identity
        or binding.get("job_index") != job["job_index"]
        or binding.get("selection_sha256") != job["selection_sha256"]
        or binding.get("trace_identity_sha256") != job["trace_identity_sha256"]
        or binding.get("trace_manifest_file_sha256") != job["trace_manifest_file_sha256"]
        or binding.get("trace_completion_file_sha256") != job["trace_completion_file_sha256"]
        or binding.get("trace_npz_sha256") != job["trace_npz_sha256"]
        or binding.get("physical_attempt_to_anonymous_slot")
        != job["physical_attempt_to_anonymous_slot"]
        or binding.get("hash_random_control_attempt") != job["hash_random_control_attempt"]
        or input_trace.get("identity_sha256") != job["trace_identity_sha256"]
        or input_trace.get("manifest_sha256") != job["trace_manifest_file_sha256"]
        or input_trace.get("completion_sha256") != job["trace_completion_file_sha256"]
        or input_trace.get("trace_npz_sha256") != job["trace_npz_sha256"]
        or target.get("global_seed") != job["global_seed"]
        or target.get("class_id") != job["class_id"]
        or target.get("slot") != job["class_slot"]
        or rollback.get("sampling_step_index_zero_based") != 149
        or rollback.get("internal_timestep") != 100
        or rollback.get("suffix_transition_count_including_t0") != 101
        or rollback.get("stochastic_transition_count") != 100
        or streams != job["fresh_stream_seeds"]
    ):
        raise RuntimeError(f"suffix output contract changed: {root}")
    if (
        receipt.get("job_index") != job["job_index"]
        or receipt.get("outdir") != str(root)
        or receipt.get("manifest_identity_sha256") != manifest["identity_sha256"]
        or receipt.get("manifest_file_sha256") != sha256_file(manifest_path)
        or receipt.get("completion_payload_sha256") != completion["payload_sha256"]
        or receipt.get("completion_file_sha256") != sha256_file(completion_path)
    ):
        raise RuntimeError(f"receipt/output binding changed: {root}")
    if (
        results.get("selection_performed") is not False
        or results.get("selected_attempt") is not None
        or results.get("quality_scores_or_features_computed") is not False
        or results.get("manifest_identity_sha256") != manifest["identity_sha256"]
        or results.get("branch_count") != BRANCH_COUNT
        or results.get("fresh_attempt_count") != len(FRESH_ATTEMPTS)
        or completion.get("complete") is not True
        or completion.get("manifest_identity_sha256") != manifest["identity_sha256"]
        or completion.get("manifest_file_sha256") != sha256_file(manifest_path)
        or completion.get("results_payload_sha256") != results["payload_sha256"]
        or completion.get("results_file_sha256") != sha256_file(results_path)
        or completion.get("branch_count") != BRANCH_COUNT
    ):
        raise RuntimeError(f"suffix results/completion changed: {root}")
    runner_snapshot = root / "runner_source.py"
    if sha256_file(runner_snapshot) != runner_sha:
        raise RuntimeError(f"runner snapshot changed: {root}")
    branches = results.get("branches")
    if not isinstance(branches, list) or [row.get("attempt_index") for row in branches] != list(range(BRANCH_COUNT)):
        raise RuntimeError(f"suffix branch axis changed: {root}")
    if (
        results.get("branches_sha256") != canonical_sha256(branches)
        or completion.get("branches_sha256") != results["branches_sha256"]
    ):
        raise RuntimeError(f"suffix branch aggregate changed: {root}")
    expected_files = {manifest_path, results_path, completion_path, runner_snapshot}
    records: dict[int, dict[str, Any]] = {}
    for attempt, summary in enumerate(branches):
        branch_name = f"attempt_{attempt:03d}"
        branch_json_path = root / "branches" / branch_name / "branch.json"
        branch = load_self_hashed(branch_json_path, "payload_sha256")
        png_record = branch.get("target_png", {})
        trace_record = branch.get("trace_npz", {})
        png_path = path_within(root, png_record.get("relative_path"))
        trace_path = path_within(root, trace_record.get("relative_path"))
        expected_seed = None if attempt == 0 else job["fresh_stream_seeds"][attempt - 1]
        expected_role = "exact_baseline_replay" if attempt == 0 else "fresh_target_suffix"
        if (
            not isinstance(summary, dict)
            or summary.get("branch") != branch_name
            or summary.get("attempt_index") != attempt
            or branch.get("branch") != branch_name
            or branch.get("attempt_index") != attempt
            or branch.get("role") != expected_role
            or branch.get("stream_seed") != expected_seed
            or branch.get("transition_count") != 101
            or branch.get("fresh_full_2b_draw_count") != (0 if attempt == 0 else 101)
            or summary.get("branch_json_sha256") != sha256_file(branch_json_path)
            or summary.get("branch_payload_sha256") != branch["payload_sha256"]
            or summary.get("target_png") != png_record
            or summary.get("trace_npz_sha256") != trace_record.get("sha256")
        ):
            raise RuntimeError(f"branch binding changed: {root}/{branch_name}")
        if (
            trace_path.is_symlink()
            or not trace_path.is_file()
            or trace_path.stat().st_size != trace_record.get("bytes")
            or sha256_file(trace_path) != trace_record.get("sha256")
        ):
            raise RuntimeError(f"branch trace changed: {trace_path}")
        records[attempt] = {
            "path": png_path,
            "record": dict(png_record),
            "branch_json_sha256": sha256_file(branch_json_path),
        }
        expected_files.update({branch_json_path, png_path, trace_path})
    actual_files = {path.resolve() for path in root.rglob("*") if path.is_file()}
    if actual_files != {path.resolve() for path in expected_files}:
        raise RuntimeError(f"suffix output exact tree changed: {root}")
    expected_directories = {
        (root / "branches").resolve(),
        *((root / "branches" / f"attempt_{attempt:03d}").resolve() for attempt in range(BRANCH_COUNT)),
    }
    actual_directories = {path.resolve() for path in root.rglob("*") if path.is_dir()}
    if actual_directories != expected_directories:
        raise RuntimeError(f"suffix output directory tree changed: {root}")
    return records


def absolute_rubric() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "task": "Judge every opaque image independently at native 256x256 and enlarged view. Do not use the current batch prevalence or mean to move the threshold.",
        "severity": {
            "0": "Ordinary for this frozen model. Mild common roughness, mild oddness, stylization, and ordinary model limitations are allowed.",
            "1": "Obviously below the frozen model's ordinary level because of visible blur, melting/fusion, displacement, or structural misalignment.",
            "2": "Severe or catastrophic visible failure: major blur/fusion, broken topology, impossible attachment, severe displacement, or comparably conspicuous corruption.",
        },
        "blur_fusion": "yes | no | uncertain; conspicuous blur, melting, smearing, or fused boundaries",
        "topology_misalignment": "yes | no | uncertain; malformed, duplicated, missing, displaced, or misattached anatomy/object structure",
        "valid": "yes | no; no only when the image cannot be judged because the file/item is invalid",
        "localized_reason": "Required concise visible-pixel reason; never infer production provenance.",
        "qualification": "Seven opaque hidden qualification items use the same severity scale. A reviewer qualifies only with at least 6 of 7 exact severity answers.",
        "forbidden": [
            "production role, job, attempt, seed, slot, or internal selector",
            "another reviewer's response or any prior visual label",
            "FID, embedding, feature distance, or any batch-level metric",
        ],
    }
    payload["identity_sha256"] = self_hash(payload, "identity_sha256")
    return payload


def preservation_rubric() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "task": "Compare opaque LEFT and RIGHT endpoints only by visible pixels. Neither side's production role is disclosed.",
        "preserved": "yes when class/identity, object count, main pose, and composition are materially preserved across sides; no when materially changed; uncertain only when genuinely unjudgeable",
        "preferred_side": "left | right | tie by overall visible quality",
        "valid": "yes | no; no only for a broken or unjudgeable review item",
        "reason": "Required concise visible-pixel reason.",
        "viewing": "Inspect the pair item and, when needed, each side at native resolution.",
    }
    payload["identity_sha256"] = self_hash(payload, "identity_sha256")
    return payload


def write_csv_template(path: Path, columns: Sequence[str], ids: Sequence[str], id_field: str) -> None:
    if columns[0] != id_field:
        raise ValueError("template ID field must be the first column")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for opaque_id in ids:
            writer.writerow({id_field: opaque_id})
        handle.flush()
        os.fsync(handle.fileno())


def copy_native(source: Path, destination: Path, expected_sha: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o644)
    try:
        with source.open("rb") as reader, os.fdopen(os.dup(descriptor), "wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
    finally:
        os.close(descriptor)
    if sha256_file(destination) != expected_sha:
        raise RuntimeError("native copy differs from authenticated source PNG")


def render_image_sheet(items: Sequence[tuple[str, Path]], output: Path) -> None:
    if len(items) != 16:
        raise ValueError("absolute sheets require exactly 4x4 images")
    cell_height = 278
    canvas = Image.new("RGB", (4 * 256, 4 * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (opaque_id, path) in enumerate(items):
        with Image.open(path) as image:
            image.load()
            canvas.paste(image.convert("RGB"), ((index % 4) * 256, (index // 4) * cell_height))
        draw.text(((index % 4) * 256 + 4, (index // 4) * cell_height + 258), opaque_id, fill="black")
    canvas.save(output, format="PNG")


def render_pair(left: Path, right: Path, pair_id: str, output: Path) -> None:
    canvas = Image.new("RGB", (520, 282), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate((left, right)):
        with Image.open(path) as image:
            image.load()
            canvas.paste(image.convert("RGB"), (0 if index == 0 else 264, 0))
    draw.text((4, 261), f"{pair_id}  LEFT", fill="black")
    draw.text((268, 261), "RIGHT", fill="black")
    canvas.save(output, format="PNG")


def render_pair_sheet(items: Sequence[Path], output: Path) -> None:
    if len(items) != 4:
        raise ValueError("pair sheets require exactly four pairs")
    canvas = Image.new("RGB", (1040, 564), "white")
    for index, path in enumerate(items):
        with Image.open(path) as image:
            image.load()
            canvas.paste(image.convert("RGB"), ((index % 2) * 520, (index // 2) * 282))
    canvas.save(output, format="PNG")


def render_qualification_sheet(items: Sequence[tuple[str, Path]], output: Path) -> None:
    if len(items) != QUALIFICATION_COUNT:
        raise ValueError("qualification sheet requires seven items")
    cell_height = 278
    canvas = Image.new("RGB", (4 * 256, 2 * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (opaque_id, path) in enumerate(items):
        with Image.open(path) as image:
            image.load()
            canvas.paste(image.convert("RGB"), ((index % 4) * 256, (index // 4) * cell_height))
        draw.text(((index % 4) * 256 + 4, (index // 4) * cell_height + 258), opaque_id, fill="black")
    canvas.save(output, format="PNG")


def atomic_install_directory_noreplace(source: Path, target: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace publication requires Linux renameat2")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(target), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(f"refusing to overwrite: {target}")
        raise OSError(error, os.strerror(error), target)


def public_readme() -> str:
    return """Blind review delivery

Complete your own reviewer_N templates only. Inspect absolute/native images at
100% and enlarged view; sheets are navigation aids. Pair production roles,
source lineage, internal method assignments, and qualification gold labels are
deliberately absent. Do not consult another reviewer or any external metric.
Qualification requires at least 6/7 exact severity answers.
    """


def validate_public_lineage_absence(root: Path, forbidden_values: Sequence[str]) -> None:
    """Fail closed if any public text file contains a private identity or path."""
    forbidden = {value for value in forbidden_values if value}
    forbidden.update({"/data/", "/home/", "class207"})
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".csv", ".txt"}:
            continue
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"public text file is not UTF-8: {path}") from exc
        leaked = sorted(value for value in forbidden if value in text)
        if leaked:
            raise RuntimeError(f"private lineage leaked into public file {path}: {leaked!r}")


def publish(args: argparse.Namespace) -> None:
    lock = resolve_input_root(args.lock, "prospective lock")
    internal_root = resolve_input_root(args.internal_product, "internal product")
    anchor_root = resolve_input_root(args.anchor_pack, "qualification anchor pack")
    # Preserve the requested leaf name so a dangling symlink cannot redirect a publication.
    delivery = args.delivery.expanduser().absolute()
    private = args.private.expanduser().absolute()
    if delivery == private or delivery in private.parents or private in delivery.parents:
        raise RuntimeError("public delivery and private mapping must be separate trees")
    if os.path.lexists(delivery) or os.path.lexists(private):
        raise RuntimeError("refusing to overwrite blind-review artifacts")

    # Phase 1: no prospective endpoint PNG may be opened before all three gates pass.
    lock_manifest, protocol, _ = validate_lock(lock)
    receipts, receipt_seals = validate_receipts(protocol, lock_manifest["identity_sha256"])
    internal_manifest, selection_rows, selection_sha = validate_internal_product(
        internal_root,
        lock=lock,
        lock_identity=lock_manifest["identity_sha256"],
        protocol_identity=protocol["identity_sha256"],
    )
    selections = validate_selection_rows(protocol["jobs"], selection_rows)

    # Fixed old anchors are validated only after the prospective gates above.
    anchor_manifest, anchor_rows = validate_anchor_pack(anchor_root)

    forbidden_inputs = [lock, internal_root, anchor_root, Path(protocol["outputs"]["receipt_root"])]
    forbidden_inputs.extend(Path(job["outdir"]) for job in protocol["jobs"])
    for output in (delivery, private):
        if any(source == output or source in output.parents or output in source.parents for source in forbidden_inputs):
            raise RuntimeError("blind-review output overlaps an authenticated input tree")

    runner_sha = sha256_file(lock / "sources/intervene_dit_v22_transient_escape_suffix.py")
    # Phase 2: validate every nonvisual bundle binding and branch trace before PNG access.
    endpoint_records: dict[tuple[int, int], dict[str, Any]] = {}
    for job in protocol["jobs"]:
        job_index = int(job["job_index"])
        records = validate_output_metadata(
            job,
            receipts[job_index],
            lock_identity=lock_manifest["identity_sha256"],
            protocol_identity=protocol["identity_sha256"],
            runner_sha=runner_sha,
        )
        for attempt, record in records.items():
            endpoint_records[(job_index, attempt)] = record
    if len(endpoint_records) != ABSOLUTE_COUNT:
        raise RuntimeError("expected exactly 640 authenticated endpoint records")

    jobs_by_index = {int(job["job_index"]): job for job in protocol["jobs"]}
    absolute_keys = sorted(
        endpoint_records,
        key=lambda key: hidden_key("absolute.order", key[0], key[1]),
    )
    absolute_id_by_key = {
        key: f"I{ordinal:04d}" for ordinal, key in enumerate(absolute_keys)
    }
    pair_keys = sorted(
        ((int(job["job_index"]), attempt) for job in protocol["jobs"] for attempt in FRESH_ATTEMPTS),
        key=lambda key: hidden_key("pair.order", key[0], key[1]),
    )
    qualification_rows = sorted(
        anchor_rows,
        key=lambda row: hidden_key("qualification.order", row["anchor_id"]),
    )

    delivery.parent.mkdir(parents=True, exist_ok=True)
    private.parent.mkdir(parents=True, exist_ok=True)
    delivery_stage = Path(tempfile.mkdtemp(prefix=f".{delivery.name}.tmp-", dir=delivery.parent))
    private_stage = Path(tempfile.mkdtemp(prefix=f".{private.name}.tmp-", dir=private.parent))
    installed_delivery = False
    installed_private = False
    try:
        absolute_native = delivery_stage / "absolute" / "native"
        absolute_sheets = delivery_stage / "absolute" / "sheets"
        pair_items = delivery_stage / "preservation" / "pairs"
        pair_sheets = delivery_stage / "preservation" / "sheets"
        qualification_native = delivery_stage / "qualification" / "native"
        qualification_sheets = delivery_stage / "qualification" / "sheets"
        templates = delivery_stage / "templates"
        for directory in (
            absolute_native,
            absolute_sheets,
            pair_items,
            pair_sheets,
            qualification_native,
            qualification_sheets,
            templates,
        ):
            directory.mkdir(parents=True, exist_ok=False)

        # Phase 3: endpoint PNG access begins only here, after all gates and metadata.
        source_pngs: list[dict[str, Any]] = []
        absolute_mapping: list[dict[str, Any]] = []
        for key in absolute_keys:
            job_index, attempt = key
            job = jobs_by_index[job_index]
            selection = selections[job_index]
            source = endpoint_records[key]["path"]
            record = endpoint_records[key]["record"]
            file_hash, pixel_hash = validate_png(source, record)
            opaque_id = absolute_id_by_key[key]
            destination = absolute_native / f"{opaque_id}.png"
            copy_native(source, destination, file_hash)
            slot = "baseline_replay" if attempt == 0 else job["physical_attempt_to_anonymous_slot"][str(attempt)]
            method_roles = []
            if attempt > 0:
                if attempt == selection["max_attempt"]:
                    method_roles.append("h10_max_nonconformity_candidate")
                if attempt == selection["medoid_attempt"]:
                    method_roles.append("h10_medoid_negative_control")
                if attempt == selection["hash_random_attempt"]:
                    method_roles.append("frozen_hash_random_control")
            source_pngs.append(
                {
                    "job_index": job_index,
                    "attempt": attempt,
                    "source_path": str(source),
                    "source_file_sha256": file_hash,
                    "source_pixel_rgb_sha256": pixel_hash,
                    "branch_json_sha256": endpoint_records[key]["branch_json_sha256"],
                }
            )
            absolute_mapping.append(
                {
                    "image_id": opaque_id,
                    "job_index": job_index,
                    "global_seed": job["global_seed"],
                    "class_id": job["class_id"],
                    "class_slot": job["class_slot"],
                    "attempt": attempt,
                    "anonymous_method_slot": slot,
                    "method_roles": method_roles,
                    "source_file_sha256": file_hash,
                    "public_native_sha256": sha256_file(destination),
                }
            )
        for start in range(0, ABSOLUTE_COUNT, 16):
            block = absolute_mapping[start : start + 16]
            render_image_sheet(
                [(row["image_id"], absolute_native / f"{row['image_id']}.png") for row in block],
                absolute_sheets / f"sheet_{start // 16:03d}.png",
            )

        pair_mapping: list[dict[str, Any]] = []
        for ordinal, (job_index, fresh_attempt) in enumerate(pair_keys):
            pair_id = f"P{ordinal:04d}"
            baseline_key = (job_index, 0)
            fresh_key = (job_index, fresh_attempt)
            left_is_baseline = int(hidden_key("pair.side", job_index, fresh_attempt)[:16], 16) % 2 == 0
            left_key, right_key = (
                (baseline_key, fresh_key) if left_is_baseline else (fresh_key, baseline_key)
            )
            left_id = absolute_id_by_key[left_key]
            right_id = absolute_id_by_key[right_key]
            output = pair_items / f"{pair_id}.png"
            render_pair(
                absolute_native / f"{left_id}.png",
                absolute_native / f"{right_id}.png",
                pair_id,
                output,
            )
            selection = selections[job_index]
            pair_mapping.append(
                {
                    "pair_id": pair_id,
                    "job_index": job_index,
                    "fresh_attempt": fresh_attempt,
                    "fresh_anonymous_method_slot": jobs_by_index[job_index]["physical_attempt_to_anonymous_slot"][str(fresh_attempt)],
                    "fresh_method_roles": [
                        role
                        for role, selected_attempt in (
                            ("h10_max_nonconformity_candidate", selection["max_attempt"]),
                            ("h10_medoid_negative_control", selection["medoid_attempt"]),
                            ("frozen_hash_random_control", selection["hash_random_attempt"]),
                        )
                        if fresh_attempt == selected_attempt
                    ],
                    "left_attempt": 0 if left_is_baseline else fresh_attempt,
                    "right_attempt": fresh_attempt if left_is_baseline else 0,
                    "left_role": "attempt0_replay" if left_is_baseline else "fresh_suffix",
                    "right_role": "fresh_suffix" if left_is_baseline else "attempt0_replay",
                    "left_absolute_image_id": left_id,
                    "right_absolute_image_id": right_id,
                    "left_source_file_sha256": endpoint_records[left_key]["record"]["sha256"],
                    "right_source_file_sha256": endpoint_records[right_key]["record"]["sha256"],
                    "public_pair_sha256": sha256_file(output),
                }
            )
        for start in range(0, PAIR_COUNT, 4):
            block = pair_mapping[start : start + 4]
            render_pair_sheet(
                [pair_items / f"{row['pair_id']}.png" for row in block],
                pair_sheets / f"sheet_{start // 4:03d}.png",
            )

        qualification_mapping: list[dict[str, Any]] = []
        qualification_public: list[tuple[str, Path]] = []
        for ordinal, row in enumerate(qualification_rows):
            qualification_id = f"Q{ordinal:03d}"
            image_record = row.get("image", {})
            source = path_within(anchor_root, image_record.get("relative_path"))
            file_hash, pixel_hash = validate_png(source, image_record)
            destination = qualification_native / f"{qualification_id}.png"
            copy_native(source, destination, file_hash)
            category = str(row.get("category"))
            gold_severity = 0 if category == "ordinary" else 2 if category.startswith("clear_bad") else -1
            if gold_severity not in {0, 2}:
                raise RuntimeError("unsupported anchor gold category")
            qualification_public.append((qualification_id, destination))
            qualification_mapping.append(
                {
                    "qualification_id": qualification_id,
                    "source_anchor_id": row["anchor_id"],
                    "source_category": category,
                    "gold_severity": gold_severity,
                    "source_path": str(source),
                    "source_file_sha256": file_hash,
                    "source_pixel_rgb_sha256": pixel_hash,
                    "public_native_sha256": sha256_file(destination),
                }
            )
        render_qualification_sheet(
            qualification_public,
            qualification_sheets / "qualification_sheet.png",
        )

        absolute_ids = [row["image_id"] for row in absolute_mapping]
        pair_ids = [row["pair_id"] for row in pair_mapping]
        qualification_ids = [row["qualification_id"] for row in qualification_mapping]
        for reviewer in range(1, REVIEWER_COUNT + 1):
            write_csv_template(
                templates / f"reviewer_{reviewer}_absolute.csv",
                ABSOLUTE_COLUMNS,
                absolute_ids,
                "image_id",
            )
            write_csv_template(
                templates / f"reviewer_{reviewer}_preservation.csv",
                PRESERVATION_COLUMNS,
                pair_ids,
                "pair_id",
            )
            write_csv_template(
                templates / f"reviewer_{reviewer}_qualification.csv",
                QUALIFICATION_COLUMNS,
                qualification_ids,
                "qualification_id",
            )
        write_json(delivery_stage / "absolute_rubric.json", absolute_rubric())
        write_json(delivery_stage / "preservation_rubric.json", preservation_rubric())
        write_bytes(delivery_stage / "README.txt", public_readme().encode("utf-8"))

        public_files = exact_file_records(delivery_stage, excluded=set())
        public_manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": PUBLIC_KIND,
            "status": "complete",
            "absolute_image_count": ABSOLUTE_COUNT,
            "absolute_sheet_count": ABSOLUTE_COUNT // 16,
            "preservation_pair_count": PAIR_COUNT,
            "preservation_sheet_count": PAIR_COUNT // 4,
            "qualification_item_count": QUALIFICATION_COUNT,
            "qualification_required_correct": 6,
            "reviewer_count": REVIEWER_COUNT,
            "response_template_count": REVIEWER_COUNT * 3,
            "private_mapping_present": False,
            "source_lineage_present": False,
            "internal_method_assignment_present": False,
            "external_metric_or_prior_label_present": False,
            "randomization_namespace_sha256": hashlib.sha256(NAMESPACE.encode("utf-8")).hexdigest(),
            "file_count_excluding_manifest": len(public_files),
            "files": public_files,
        }
        public_manifest["identity_sha256"] = self_hash(public_manifest, "identity_sha256")
        write_json(delivery_stage / "manifest.json", public_manifest)
        validate_public_lineage_absence(
            delivery_stage,
            (
                str(lock),
                str(internal_root),
                str(anchor_root),
                str(Path(protocol["outputs"]["receipt_root"]).expanduser().resolve()),
                lock_manifest["identity_sha256"],
                protocol["identity_sha256"],
                internal_manifest["identity_sha256"],
                anchor_manifest["identity_sha256"],
                selection_sha,
                NAMESPACE,
            ),
        )

        job_methods = []
        for job_index in range(JOB_COUNT):
            job = jobs_by_index[job_index]
            row = selections[job_index]
            job_methods.append(
                {
                    "job_index": job_index,
                    "global_seed": job["global_seed"],
                    "class_id": job["class_id"],
                    "class_slot": job["class_slot"],
                    "rollback_step": job["rollback_sampling_step"],
                    "physical_attempt_to_anonymous_slot": job["physical_attempt_to_anonymous_slot"],
                    "max_attempt": row["max_attempt"],
                    "max_slot": row["max_slot"],
                    "medoid_attempt": row["medoid_attempt"],
                    "medoid_slot": row["medoid_slot"],
                    "hash_random_attempt": row["hash_random_attempt"],
                    "hash_random_slot": row["hash_random_slot"],
                }
            )
        private_payload: dict[str, Any] = {
            "schema_version": 1,
            "artifact_kind": PRIVATE_KIND,
            "status": "SEALED_UNTIL_THREE_REVIEWS_AND_QUALIFICATION_LOCKS",
            "builder_source_sha256": sha256_file(Path(__file__).resolve()),
            "public_delivery_identity_sha256": public_manifest["identity_sha256"],
            "public_manifest_file_sha256": sha256_file(delivery_stage / "manifest.json"),
            "prospective_lock_identity_sha256": lock_manifest["identity_sha256"],
            "prospective_lock_manifest_file_sha256": sha256_file(lock / "manifest.json"),
            "prospective_protocol_identity_sha256": protocol["identity_sha256"],
            "prospective_protocol_file_sha256": sha256_file(lock / "protocol.json"),
            "internal_product_identity_sha256": internal_manifest["identity_sha256"],
            "internal_product_manifest_file_sha256": sha256_file(internal_root / "manifest.json"),
            "internal_product_completion_file_sha256": sha256_file(internal_root / "completion.json"),
            "sealed_selections_file_sha256": selection_sha,
            "anchor_pack_manifest_identity_sha256": anchor_manifest["identity_sha256"],
            "anchor_pack_manifest_file_sha256": sha256_file(anchor_root / "manifest.json"),
            "anchor_pack_completion_file_sha256": sha256_file(anchor_root / "completion.json"),
            "receipt_seals": receipt_seals,
            "receipt_output_records": [receipts[index] for index in range(JOB_COUNT)],
            "randomization_namespace": NAMESPACE,
            "absolute_mapping": absolute_mapping,
            "preservation_mapping": pair_mapping,
            "qualification_mapping": qualification_mapping,
            "job_method_assignments": job_methods,
            "source_pngs": source_pngs,
            "qualification_rule": {
                "correct_required": 6,
                "total": 7,
                "exact_severity_match": True,
                "ordinary_gold": 0,
                "clear_bad_gold": 2,
            },
            "public_order_or_side_used_selector_values_images_labels_FID_or_embeddings": False,
            "only_old_anchor_gold_used": True,
        }
        private_payload["identity_sha256"] = self_hash(private_payload, "identity_sha256")
        write_json(private_stage / "sealed_mapping.json", private_payload)

        # Recheck public exact tree and private self hash before publication.
        observed_public = load_self_hashed(delivery_stage / "manifest.json", "identity_sha256")
        actual_public_files = exact_file_records(delivery_stage, excluded={"manifest.json"})
        if observed_public != public_manifest or actual_public_files != public_files:
            raise RuntimeError("public delivery changed before publication")
        if load_self_hashed(private_stage / "sealed_mapping.json", "identity_sha256") != private_payload:
            raise RuntimeError("private mapping changed before publication")
        if exact_file_records(private_stage, excluded=set()) != [
            {
                "name": "sealed_mapping.json",
                "bytes": (private_stage / "sealed_mapping.json").stat().st_size,
                "sha256": sha256_file(private_stage / "sealed_mapping.json"),
            }
        ]:
            raise RuntimeError("private mapping exact tree changed before publication")

        atomic_install_directory_noreplace(private_stage, private)
        installed_private = True
        atomic_install_directory_noreplace(delivery_stage, delivery)
        installed_delivery = True
    except BaseException:
        if delivery_stage.exists():
            shutil.rmtree(delivery_stage, ignore_errors=True)
        if private_stage.exists():
            shutil.rmtree(private_stage, ignore_errors=True)
        # The only possible partial publication is a just-created private tree.
        if installed_private and not installed_delivery and private.exists():
            shutil.rmtree(private, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "complete",
                "delivery": str(delivery),
                "private": str(private),
                "delivery_identity_sha256": public_manifest["identity_sha256"],
                "private_identity_sha256": private_payload["identity_sha256"],
                "absolute_images": ABSOLUTE_COUNT,
                "preservation_pairs": PAIR_COUNT,
                "qualification_items": QUALIFICATION_COUNT,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def self_test() -> None:
    absolute = [(job, attempt) for job in range(JOB_COUNT) for attempt in range(BRANCH_COUNT)]
    ordered_absolute = sorted(absolute, key=lambda key: hidden_key("absolute.order", *key))
    pairs = [(job, attempt) for job in range(JOB_COUNT) for attempt in FRESH_ATTEMPTS]
    ordered_pairs = sorted(pairs, key=lambda key: hidden_key("pair.order", *key))
    if (
        len(ordered_absolute) != ABSOLUTE_COUNT
        or len(set(ordered_absolute)) != ABSOLUTE_COUNT
        or len(ordered_pairs) != PAIR_COUNT
        or len(set(ordered_pairs)) != PAIR_COUNT
    ):
        raise AssertionError("deterministic item order is not bijective")
    left_baseline_flags = [
        int(hidden_key("pair.side", *key)[:16], 16) % 2 == 0 for key in pairs
    ]
    if not 180 <= sum(left_baseline_flags) <= 332:
        raise AssertionError("deterministic pair sides are pathologically imbalanced")
    if any(
        {0, fresh_attempt}
        != {
            0 if left_is_baseline else fresh_attempt,
            fresh_attempt if left_is_baseline else 0,
        }
        for (job_index, fresh_attempt), left_is_baseline in zip(pairs, left_baseline_flags)
    ):
        raise AssertionError("pair left/right attempt mapping changed")
    if any(
        sorted(attempt for current_job, attempt in pairs if current_job == job_index)
        != list(FRESH_ATTEMPTS)
        for job_index in range(JOB_COUNT)
    ):
        raise AssertionError("pair axis is not exactly attempt0 versus each of four fresh attempts")
    if (
        ABSOLUTE_COLUMNS
        != ("image_id", "severity", "blur_fusion", "topology_misalignment", "valid", "localized_reason")
        or PRESERVATION_COLUMNS
        != ("pair_id", "preserved", "preferred_side", "valid", "reason")
        or QUALIFICATION_COLUMNS != ("qualification_id", "severity", "valid", "reason")
    ):
        raise AssertionError("review template schema changed")
    payload = {"kind": "synthetic"}
    payload["identity_sha256"] = self_hash(payload, "identity_sha256")
    if payload["identity_sha256"] != self_hash(payload, "identity_sha256"):
        raise AssertionError("canonical self hash changed")
    if 0 != (0 if "ordinary" == "ordinary" else -1) or 2 != (2 if "clear_bad_blur".startswith("clear_bad") else -1):
        raise AssertionError("qualification gold mapping changed")
    with tempfile.TemporaryDirectory(prefix="transient-escape-blind-selftest-") as temporary:
        root = Path(temporary)
        absolute_template = root / "absolute.csv"
        preservation_template = root / "preservation.csv"
        qualification_template = root / "qualification.csv"
        write_csv_template(
            absolute_template,
            ABSOLUTE_COLUMNS,
            [f"I{index:04d}" for index in range(ABSOLUTE_COUNT)],
            "image_id",
        )
        write_csv_template(
            preservation_template,
            PRESERVATION_COLUMNS,
            [f"P{index:04d}" for index in range(PAIR_COUNT)],
            "pair_id",
        )
        write_csv_template(
            qualification_template,
            QUALIFICATION_COLUMNS,
            [f"Q{index:03d}" for index in range(QUALIFICATION_COUNT)],
            "qualification_id",
        )
        for path, columns, expected_rows in (
            (absolute_template, ABSOLUTE_COLUMNS, ABSOLUTE_COUNT),
            (preservation_template, PRESERVATION_COLUMNS, PAIR_COUNT),
            (qualification_template, QUALIFICATION_COLUMNS, QUALIFICATION_COUNT),
        ):
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            if tuple(reader.fieldnames or ()) != tuple(columns) or len(rows) != expected_rows:
                raise AssertionError("synthetic review template schema/count changed")
        validate_public_lineage_absence(root, ("synthetic-private-id",))
        write_bytes(root / "must_fail.txt", b"class207 must never be public\n")
        try:
            validate_public_lineage_absence(root, ("synthetic-private-id",))
        except RuntimeError:
            pass
        else:
            raise AssertionError("public lineage firewall failed closed")
    print("transient-escape blind-review self-test passed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--internal-product", type=Path, default=DEFAULT_INTERNAL_PRODUCT)
    parser.add_argument("--anchor-pack", type=Path, default=DEFAULT_ANCHOR_PACK)
    parser.add_argument("--delivery", type=Path)
    parser.add_argument("--private", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.delivery is None or args.private is None:
        raise SystemExit("--delivery and --private are required unless --self-test is used")
    publish(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
